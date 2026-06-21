from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0004_alter_apisnapshot_source"),
    ]

    operations = [
        migrations.AddField(
            model_name="agripredictbatch",
            name="model_name",
            field=models.CharField(default="probabilistic", max_length=32, db_index=True),
        ),
        migrations.AddIndex(
            model_name="agripredictbatch",
            index=models.Index(
                fields=["item_id", "model_name", "-created_at"],
                name="dashboard_a_item_id_96f50a_idx",
            ),
        ),
    ]
