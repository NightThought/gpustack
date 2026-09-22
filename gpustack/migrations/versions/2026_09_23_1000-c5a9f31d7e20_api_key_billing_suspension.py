"""api key billing suspension

Revision ID: c5a9f31d7e20
Revises: a7f3c9e21b40
Create Date: 2026-09-23 10:00:00.000000

Adds the two columns that let a wallet running dry reach the request path
(WP5 of ``docs/prd/13-计费引擎开发TODO.md``):

* ``api_keys.suspended`` — server-set only. A suspended key is filtered out of
  the gateway's local auth tables (``build_local_auth_tables``), so the gateway
  cannot verify it locally and forwards the request to ``/token-auth``, which
  answers 402. Excluding rather than flagging is the shape this table already
  uses for deactivated principals and soft-deleted keys, and it needs no change
  to the gateway plugin: absence from the table is the only signal the plugin
  understands.
* ``api_keys.suspension_reason`` — the tenant-facing why. Prefixed with the
  subsystem that set it (``billing:``), so clearing a billing suspension cannot
  clear one an admin imposed for another reason.

Both default to "not suspended": an upgrade must not lock anyone out, and a key
whose wallet has never been evaluated keeps working until the settler says
otherwise.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

# revision identifiers, used by Alembic.
revision: str = 'c5a9f31d7e20'
down_revision: Union[str, None] = 'a7f3c9e21b40'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'suspended', sa.Boolean(), nullable=False, server_default=sa.false()
            )
        )
        batch_op.add_column(
            sa.Column(
                'suspension_reason',
                sqlmodel.sql.sqltypes.AutoString(length=255),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.drop_column('suspension_reason')
        batch_op.drop_column('suspended')
