"""DrivAerNet surface-pressure inference app (cluster side).

Serves on-demand pressure-field prediction for arbitrary car STLs via a small
FastAPI server. Reuses the validated inference path from
``single_car_inference_fixed.py``. See ``app/server.py``.
"""
