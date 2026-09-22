"""billing tables

Revision ID: a7f3c9e21b40
Revises: f1a2b3c4d5e6
Create Date: 2026-09-22 10:00:00.000000

Creates the storage layer of the billing engine (WP1 of
``docs/prd/13-计费引擎开发TODO.md``). Eight tables, in the order the money
flows through them:

* ``billing_wallet``       — prepaid balance of one ORG principal (the
                             settlement subject). Non-negative by construction;
                             running dry sets ``suspended``, which the gateway
                             auth reconciler turns into rejected api keys.
* ``billing_price_book``   — SKU × model × effective-window prices. Windows must
                             not overlap for one (sku, model_name); that is a
                             service-layer check, not expressible in portable DDL.
* ``billing_ledger``       — the immutable charge/credit record and the system's
                             single source of truth. ``(source_table, source_id,
                             sku)`` is UNIQUE: the rater re-scans usage rows by
                             design, and this constraint is what turns a re-run
                             into a no-op instead of a double charge.
* ``billing_session``      — realtime (token) pre-consume → settle / refund state
                             machine. ``settled`` / ``refunded`` are one-way and
                             mutually exclusive, which is what makes both
                             operations safe to retry.
* ``billing_invoice`` /
  ``billing_invoice_item`` — billing-period document. Covers the DEFERRED
                             (resource) entries; issuing it debits the wallet
                             once for the total. One invoice per
                             (principal, period).
* ``billing_quota``        — spend/usage ceilings at organization / user /
                             api-key scope, read by the gateway's QuotaGate.
                             NOTE: SQL treats NULLs as distinct, so the unique
                             constraint only fully guards scopes whose subject
                             columns are all non-NULL — the service layer must
                             reject duplicates for the NULL-subject scopes.
* ``billing_redemption``   — single-use top-up codes. Redeeming is a
                             lock + ``WHERE status = 'enabled'`` CAS flip, so
                             concurrent redemption of one code charges once.

Conventions follow the neighbouring audit tables (``metered_usage``,
``model_usage_details``): attribution id columns on the ledger are FK-less plain
integers with ``*_name`` snapshots, because a charge must stay traceable after
the user, api key, model or cluster it was raised against is deleted; enums are
stored as VARCHAR rather than native database enums (matching
``cache_services.state``), so adding a value is a code change and not a
migration; every money / quantity column is ``Numeric(20, 8)``, the same scale
``metered_usage.sku_count`` uses, so a price × quantity product never loses the
fraction a sliced accelerator contributes.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

from gpustack.schemas.common import UTCDateTime

# revision identifiers, used by Alembic.
revision: str = 'a7f3c9e21b40'
down_revision: Union[str, None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Money / quantity columns: Numeric(20, 8) everywhere, matching
# gpustack.schemas.billing.MONEY_PRECISION / MONEY_SCALE.
_MONEY = sa.Numeric(precision=20, scale=8)


def _timestamps() -> list:
    """The three columns ``BaseModelMixin`` contributes to every table."""
    return [
        sa.Column('created_at', UTCDateTime(), nullable=False),
        sa.Column('updated_at', UTCDateTime(), nullable=False),
        sa.Column('deleted_at', UTCDateTime(), nullable=True),
    ]


def upgrade() -> None:
    # —— billing_wallet ————————————————————————————————————————————
    op.create_table(
        'billing_wallet',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('principal_id', sa.Integer(), nullable=False),
        sa.Column(
            'principal_name',
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=True,
        ),
        sa.Column('currency', sa.String(length=8), nullable=False),
        sa.Column('balance', _MONEY, nullable=False, server_default='0'),
        sa.Column('frozen', _MONEY, nullable=False, server_default='0'),
        sa.Column('suspended', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('suspended_at', UTCDateTime(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('principal_id', name='uq_wallet_principal'),
    )
    op.create_index(
        'ix_billing_wallet_principal_id', 'billing_wallet', ['principal_id']
    )

    # —— billing_price_book ——————————————————————————————————————————
    op.create_table(
        'billing_price_book',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('sku', sa.String(length=64), nullable=False),
        sa.Column(
            'model_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True
        ),
        sa.Column(
            'group_name', sqlmodel.sql.sqltypes.AutoString(length=128), nullable=True
        ),
        sa.Column('unit', sa.String(length=32), nullable=False),
        sa.Column('price', _MONEY, nullable=False),
        sa.Column('per_quantity', _MONEY, nullable=False, server_default='1'),
        sa.Column('currency', sa.String(length=8), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('effective_from', UTCDateTime(), nullable=False),
        sa.Column('effective_to', UTCDateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        *_timestamps(),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_billing_price_book_sku', 'billing_price_book', ['sku'])
    op.create_index(
        'ix_billing_price_book_model_name', 'billing_price_book', ['model_name']
    )

    # —— billing_ledger ——————————————————————————————————————————————
    op.create_table(
        'billing_ledger',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        # Provenance / idempotency key.
        sa.Column('source_table', sa.String(length=64), nullable=False),
        sa.Column('source_id', sa.BigInteger(), nullable=False),
        # Attribution snapshots — FK-less on purpose (see module docstring).
        sa.Column('principal_id', sa.Integer(), nullable=True),
        sa.Column(
            'principal_name',
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=True,
        ),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column(
            'user_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True
        ),
        sa.Column('api_key_id', sa.Integer(), nullable=True),
        sa.Column(
            'api_key_name',
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=True,
        ),
        sa.Column(
            'model_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True
        ),
        sa.Column('cluster_id', sa.Integer(), nullable=True),
        sa.Column('resource_id', sa.Integer(), nullable=True),
        sa.Column(
            'resource_name',
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=True,
        ),
        sa.Column(
            'request_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True
        ),
        # Pricing snapshot.
        sa.Column('sku', sa.String(length=64), nullable=False),
        sa.Column('quantity', _MONEY, nullable=False),
        sa.Column('unit', sa.String(length=32), nullable=False),
        sa.Column('unit_price', _MONEY, nullable=False),
        sa.Column('price_book_id', sa.Integer(), nullable=True),
        sa.Column('price_book_version', sa.Integer(), nullable=True),
        sa.Column('amount', _MONEY, nullable=False),
        sa.Column('currency', sa.String(length=8), nullable=False),
        sa.Column('direction', sa.String(length=16), nullable=False),
        # Settlement.
        sa.Column('settle_mode', sa.String(length=16), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('billing_session_id', sa.Integer(), nullable=True),
        sa.Column('invoice_id', sa.Integer(), nullable=True),
        sa.Column('settled_at', UTCDateTime(), nullable=True),
        sa.Column('occurred_at', UTCDateTime(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'source_table', 'source_id', 'sku', name='uq_ledger_source_sku'
        ),
    )
    op.create_index('ix_billing_ledger_sku', 'billing_ledger', ['sku'])
    op.create_index('ix_billing_ledger_request_id', 'billing_ledger', ['request_id'])

    # —— billing_session —————————————————————————————————————————————
    op.create_table(
        'billing_session',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('principal_id', sa.Integer(), nullable=True),
        sa.Column(
            'request_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True
        ),
        sa.Column('estimate', _MONEY, nullable=False, server_default='0'),
        sa.Column('actual', _MONEY, nullable=True),
        sa.Column('settled', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('refunded', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('settled_at', UTCDateTime(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_billing_session_request_id', 'billing_session', ['request_id']
    )

    # —— billing_invoice —————————————————————————————————————————————
    op.create_table(
        'billing_invoice',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('principal_id', sa.Integer(), nullable=False),
        sa.Column(
            'principal_name',
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=True,
        ),
        sa.Column('period_start', UTCDateTime(), nullable=False),
        sa.Column('period_end', UTCDateTime(), nullable=False),
        sa.Column('amount', _MONEY, nullable=False),
        sa.Column('currency', sa.String(length=8), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('issued_at', UTCDateTime(), nullable=True),
        sa.Column('settled_at', UTCDateTime(), nullable=True),
        sa.Column(
            'unpaid_reason',
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=True,
        ),
        *_timestamps(),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'principal_id', 'period_start', name='uq_invoice_principal_period'
        ),
    )
    op.create_index(
        'ix_billing_invoice_principal_id', 'billing_invoice', ['principal_id']
    )

    # —— billing_invoice_item ————————————————————————————————————————
    op.create_table(
        'billing_invoice_item',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('invoice_id', sa.Integer(), nullable=False),
        sa.Column('sku', sa.String(length=64), nullable=False),
        sa.Column(
            'model_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True
        ),
        sa.Column('quantity', _MONEY, nullable=False),
        sa.Column('unit', sa.String(length=32), nullable=False),
        sa.Column('amount', _MONEY, nullable=False),
        sa.Column('entry_count', sa.Integer(), nullable=False, server_default='0'),
        *_timestamps(),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_billing_invoice_item_invoice_id', 'billing_invoice_item', ['invoice_id']
    )

    # —— billing_quota ———————————————————————————————————————————————
    op.create_table(
        'billing_quota',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('scope', sa.String(length=16), nullable=False),
        sa.Column('principal_id', sa.Integer(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('api_key_id', sa.Integer(), nullable=True),
        sa.Column(
            'model_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True
        ),
        sa.Column('limit_type', sa.String(length=32), nullable=False),
        sa.Column('limit_value', _MONEY, nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('window_start', UTCDateTime(), nullable=True),
        sa.Column('used', _MONEY, nullable=False, server_default='0'),
        *_timestamps(),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'scope',
            'principal_id',
            'user_id',
            'api_key_id',
            'model_name',
            'limit_type',
            name='uq_quota_scope_limit',
        ),
    )

    # —— billing_redemption ——————————————————————————————————————————
    op.create_table(
        'billing_redemption',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('code', sa.String(length=64), nullable=False),
        sa.Column('amount', _MONEY, nullable=False),
        sa.Column('currency', sa.String(length=8), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column(
            'batch', sqlmodel.sql.sqltypes.AutoString(length=128), nullable=True
        ),
        sa.Column('expires_at', UTCDateTime(), nullable=True),
        sa.Column('used_by_principal_id', sa.Integer(), nullable=True),
        sa.Column('used_by_user_id', sa.Integer(), nullable=True),
        sa.Column('used_at', UTCDateTime(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code', name='uq_billing_redemption_code'),
    )


def downgrade() -> None:
    # Reverse creation order: children before parents, indexes with their table.
    op.drop_table('billing_redemption')
    op.drop_table('billing_quota')
    op.drop_index('ix_billing_invoice_item_invoice_id', 'billing_invoice_item')
    op.drop_table('billing_invoice_item')
    op.drop_index('ix_billing_invoice_principal_id', 'billing_invoice')
    op.drop_table('billing_invoice')
    op.drop_index('ix_billing_session_request_id', 'billing_session')
    op.drop_table('billing_session')
    op.drop_index('ix_billing_ledger_request_id', 'billing_ledger')
    op.drop_index('ix_billing_ledger_sku', 'billing_ledger')
    op.drop_table('billing_ledger')
    op.drop_index('ix_billing_price_book_model_name', 'billing_price_book')
    op.drop_index('ix_billing_price_book_sku', 'billing_price_book')
    op.drop_table('billing_price_book')
    op.drop_index('ix_billing_wallet_principal_id', 'billing_wallet')
    op.drop_table('billing_wallet')
