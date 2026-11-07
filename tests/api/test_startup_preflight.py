"""Startup must establish schema compatibility before constructing workers."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from api import main
from api.migrations import DatabaseSchemaError


class StartupPreflightTests(unittest.TestCase):
    def test_schema_failure_prevents_both_worker_managers_from_starting(self):
        app = SimpleNamespace(state=SimpleNamespace())
        preparation = mock.Mock()
        processing = mock.Mock()

        async def enter_lifespan():
            async with main.lifespan(app):
                self.fail("The lifespan body must not run for an incompatible schema")

        with (
            mock.patch.object(
                main,
                "require_current_schema",
                side_effect=DatabaseSchemaError("controlled incompatible schema"),
            ),
            mock.patch.object(
                main,
                "ExplanationPreparationManager",
                return_value=preparation,
            ) as preparation_type,
            mock.patch.object(
                main,
                "ProcessingJobManager",
                return_value=processing,
            ) as processing_type,
            mock.patch.object(main, "ModalWarmup") as warmup,
        ):
            with self.assertRaisesRegex(DatabaseSchemaError, "controlled incompatible"):
                asyncio.run(enter_lifespan())

        preparation_type.assert_not_called()
        processing_type.assert_not_called()
        warmup.assert_not_called()
        preparation.start.assert_not_called()
        processing.start.assert_not_called()

    def test_lifespan_fences_processing_before_other_service_cleanup(self):
        app = SimpleNamespace(state=SimpleNamespace())
        order = []
        preparation = mock.Mock()
        processing = mock.Mock()
        warmup = mock.Mock()
        processing.shutdown.side_effect = lambda: order.append("processing")
        warmup.stop.side_effect = lambda: order.append("warmup")
        preparation.shutdown.side_effect = lambda: order.append("preparation")

        async def enter_lifespan():
            async with main.lifespan(app):
                self.assertIs(app.state.processing_jobs, processing)

        with (
            mock.patch.object(main, "require_current_schema"),
            mock.patch.object(
                main, "ExplanationPreparationManager", return_value=preparation,
            ),
            mock.patch.object(main, "ProcessingJobManager", return_value=processing),
            mock.patch.object(main, "ModalWarmup", return_value=warmup),
        ):
            asyncio.run(enter_lifespan())

        self.assertEqual(order[:3], ["processing", "preparation", "warmup"])


if __name__ == "__main__":
    unittest.main()
