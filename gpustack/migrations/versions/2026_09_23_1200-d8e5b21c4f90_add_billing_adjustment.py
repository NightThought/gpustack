"""add billing_adjustment

Revision ID: d8e5b21c4f90
Revises: c5a9f31d7e20
Create Date: 2026-09-23 12:00:00.000000

One table, for the money an operator moves by hand (WP4.5 of
``docs/prd/13-计费引擎开发TODO.md``, deferred out of WP4 because it needs an
idempotency key of its own).

Redemption codes already cover money a tenant bought: the code is the
instrument, it exists before the money moves, and it can only be spent once.
A manual correction has no instrument, so the only thing standing between a
retried HTTP request and a double credit is a key the *caller* supplies — a
ticket number, a payment reference. That is why ``idempotency_key`` is its own
unique column rather than a reuse of ``billing_redemption.code``: sharing one
namespace would let a ticket number collide with a code, and the collision
fails closed, rejecting a legitimate correction as a duplicate of an unrelated
redemption.

``amount`` is signed (positive credits, negative debits) and ``ledger_entry_id``
points at the ledger row the adjustment produced, so a wallet movement can
always be traced to the entry that explains it and back to the operator who made
it. Conventions match the other billing tables: ``Numeric(20, 8)`` money, enums
as VARCHAR, ``*_name`` snapshots for attribution that has to outlive the row it
names, and the three ``BaseModelMixin`` timestamps.

There is no update or delete path in the API. Correcting a correction is another
adjustment; an edited or removed row would leave the ledger and the wallet unable
to explain each other.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime

# revision identifiers, used by Alembic.
revision: str = 'd8e5b21c4f90'
down_revision: Union[str, None] = 'c5a9f31d7e20'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Money columns: Numeric(20, 8), matching gpustack.schemas.billing.MONEY_SCALE.
_MONEY = sa.Numeric(precision=20, scale=8)


def _timestamps() -> list:
    """The three columns ``BaseModelMixin`` contributes to every table."""
    return [
        sa.Column('created_at', UTCDateTime(), nullable=False),
        sa.Column('updated_at', UTCDateTime(), nullable=False),
        sa.Column('deleted_at', UTCDateTime(), nullable=True),
    ]


def upgrade() -> None:
    op.create_table(
        'billing_adjustment',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('principal_id', sa.Integer(), nullable=False),
        sa.Column('principal_name', sa.String(length=255), nullable=True),
        sa.Column('wallet_id', sa.Integer(), nullable=True),
        # Signed: the sign is what both the ledger row and the wallet movement
        # need, so it is stored rather than recombined from a direction column.
        sa.Column('amount', _MONEY, nullable=False),
        sa.Column(
            'currency', sa.String(length=8), nullable=False, server_default='CNY'
        ),
        sa.Column('reason', sa.String(length=512), nullable=False),
        sa.Column('operator_id', sa.Integer(), nullable=True),
        sa.Column('operator_name', sa.String(length=255), nullable=True),
        # 191 is the longest prefix MySQL indexes under utf8mb4, so the unique
        # constraint below survives a deployment on that database.
        sa.Column('idempotency_key', sa.String(length=191), nullable=False),
        sa.Column('ledger_entry_id', sa.Integer(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'idempotency_key', name='uq_adjustment_idempotency'
        ),
    )
    op.create_index(
        'ix_billing_adjustment_principal_id',
        'billing_adjustment',
        ['principal_id'],
    )


def downgrade() -> None:
    op.drop_index('ix_billing_adjustment_principal_id', table_name='billing_adjustment')
    op.drop_table('billing_adjustment')
