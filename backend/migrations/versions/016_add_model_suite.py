"""Add model_suite to settings table

Revision ID: 016_add_model_suite
Revises: 015_rename_baidu_ocr_api_key
Create Date: 2026-03-02

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '016_add_model_suite'
down_revision = '015_rename_baidu_ocr_api_key'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('settings', sa.Column('model_suite', sa.String(20), nullable=True))


def downgrade():
    op.drop_column('settings', 'model_suite')
