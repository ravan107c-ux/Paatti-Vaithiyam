import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import Mock, patch


_test_directory = tempfile.TemporaryDirectory()
os.environ["PAATTI_DB_PATH"] = os.path.join(_test_directory.name, "test-paatti.db")
os.environ["GEMINI_API_KEY"] = "test-gemini-key"
os.environ["REMEDY_REVIEW_TOKEN"] = "test-review-token"

import app as app_module


def grounded_response(relation, support_level, reason):
    return {
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            '{"relation": "%s", "support_level": "%s", "reason": "%s", '
                            '"identified_herbs": "hibiscus", "problem": "hair fall"}'
                            % (relation, support_level, reason)
                        ),
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "Ethnobotanical reference",
                                "url": "https://example.org/herb-reference",
                            }
                        ],
                    }
                ],
            }
        ]
    }


class RemedyScreeningTests(unittest.TestCase):
    counter = 0

    def setUp(self):
        self.client = app_module.app.test_client()
        app_module.app.config["TESTING"] = True
        type(self).counter += 1
        response = self.client.post(
            "/api/auth/register",
            json={
                "name": "Test User",
                "phone": f"+1555000{type(self).counter:04d}",
                "email": f"user{type(self).counter}@example.org",
                "location": "Madurai",
                "password": "safe-test-password",
            },
        )
        self.assertEqual(response.status_code, 201)

    @patch("app.requests.post")
    def test_related_grounded_remedy_is_saved_for_human_review(self, post):
        response = Mock(ok=True)
        response.json.return_value = grounded_response(
            "related", "traditional_use", "A reference describes traditional topical use."
        )
        post.return_value = response

        result = self.client.post(
            "/api/remedies",
            json={"raw_text": "Hibiscus oil is traditionally used for hair fall."},
        )

        self.assertEqual(result.status_code, 201)
        data = result.get_json()
        self.assertEqual(data["publication_status"], "pending")
        self.assertEqual(data["herbs"], ["hibiscus"])
        self.assertEqual(data["symptom"], "hair fall")
        self.assertTrue(data["gemini_screening"]["sources"])
        self.assertEqual(post.call_args.kwargs["json"]["tools"], [{"type": "google_search"}])

        review_result = self.client.get(
            "/api/review/remedies",
            headers={"X-Review-Token": "test-review-token"},
        ).get_json()
        submitted = next(item for item in review_result if item["id"] == data["id"])
        self.assertEqual(submitted["gemini_screening"]["support_level"], "traditional_use")

    @patch("app.requests.post")
    def test_unrelated_remedy_is_rejected_without_database_insert(self, post):
        response = Mock(ok=True)
        response.json.return_value = grounded_response(
            "unrelated", "no_reliable_support", "The sources do not connect the ingredient to the problem."
        )
        post.return_value = response
        with closing(sqlite3.connect(app_module.DB_PATH)) as database:
            before = database.execute(
                "SELECT COUNT(*) FROM remedies WHERE submitted_by IS NOT NULL"
            ).fetchone()[0]

        result = self.client.post(
            "/api/remedies",
            json={"raw_text": "A random unrelated herb for an unrelated concern."},
        )

        with closing(sqlite3.connect(app_module.DB_PATH)) as database:
            after = database.execute(
                "SELECT COUNT(*) FROM remedies WHERE submitted_by IS NOT NULL"
            ).fetchone()[0]
        self.assertEqual(result.status_code, 422)
        self.assertEqual(before, after)
        self.assertIn("not added", result.get_json()["error"])

    @patch("app.requests.post")
    def test_missing_grounded_sources_prevents_submission(self, post):
        response = Mock(ok=True)
        response.json.return_value = {
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '{"relation":"related","support_level":"traditional_use",'
                                '"reason":"A traditional relationship may exist."}'
                            ),
                        }
                    ],
                }
            ]
        }
        post.return_value = response

        result = self.client.post(
            "/api/remedies",
            json={"raw_text": "Hibiscus is traditionally used for hair fall."},
        )

        self.assertEqual(result.status_code, 503)
        self.assertIn("verifiable sources", result.get_json()["error"])

    @patch.dict(os.environ, {"GEMINI_API_KEY": ""})
    @patch("app.requests.post")
    def test_missing_api_key_prevents_submission(self, post):
        result = self.client.post(
            "/api/remedies",
            json={"raw_text": "Hibiscus is traditionally used for hair fall."},
        )

        self.assertEqual(result.status_code, 503)
        self.assertIn("Configure a Gemini API key", result.get_json()["error"])
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
