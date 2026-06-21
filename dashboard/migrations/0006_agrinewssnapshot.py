from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0005_agripredictbatch_model_name"),
    ]

    operations = [
        migrations.CreateModel(
            name="AgriNewsSnapshot",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("query", models.TextField()),
                ("summary", models.TextField(blank=True)),
                ("articles", models.JSONField(default=list)),
                ("raw_response", models.JSONField(default=dict)),
                ("fetched_at", models.DateTimeField(auto_now_add=True, db_index=True)),
            ],
            options={
                "ordering": ["-fetched_at"],
                "indexes": [models.Index(fields=["-fetched_at"], name="dashboard_a_fetched_4adf6e_idx")],
            },
        ),
    ]
