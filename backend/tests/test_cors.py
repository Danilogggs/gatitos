import unittest

from fastapi.testclient import TestClient

from app.main import app, frontend_origins


class CorsTest(unittest.TestCase):
    def test_local_frontend_accepts_both_loopback_names(self):
        self.assertEqual(frontend_origins('http://localhost:5173'),
                         ['http://localhost:5173', 'http://127.0.0.1:5173'])
        self.assertEqual(frontend_origins('http://127.0.0.1:5173'),
                         ['http://127.0.0.1:5173', 'http://localhost:5173'])

    def test_remote_frontend_does_not_gain_extra_origins(self):
        self.assertEqual(frontend_origins('https://cats.example.com'),
                         ['https://cats.example.com'])

    def test_preflight_allows_local_browser_but_not_unrelated_origin(self):
        client = TestClient(app)
        for origin in ('http://localhost:5173', 'http://127.0.0.1:5173'):
            response = client.options('/api/config', headers={
                'Origin': origin, 'Access-Control-Request-Method': 'GET',
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['access-control-allow-origin'], origin)
        denied = client.options('/api/config', headers={
            'Origin': 'http://unrelated.example:5173', 'Access-Control-Request-Method': 'GET',
        })
        self.assertNotIn('access-control-allow-origin', denied.headers)
