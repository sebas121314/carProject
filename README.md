# Evaluación de Vehículos - Guía rápida

Aplicación Flask que carga un modelo entrenado (dataset UCI Car Evaluation) y permite predecir la aceptabilidad de un auto vía formulario web o endpoint `/predict`.

## Cómo funciona
- El modelo se busca en `car+evaluation/` (`best_car_model_sprint3.pkl` preferido, `best_car_model.pkl` de respaldo).
- El pipeline recibe 6 características categóricas: `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`.
- Backend (`app.py`) valida valores, arma un DataFrame y devuelve la predicción en JSON (`prediction` en español y `prediction_code` original).
- Frontend (`templates/index.html` + `static/styles.css`) muestra el formulario con campos traducidos y consume `/predict` vía fetch.

## Estructura de carpetas
- `app.py` — API Flask y render de la UI.
- `templates/index.html` — formulario de captura y lógica JS de envío.
- `static/styles.css` — estilos del formulario.
- `car+evaluation/` — dataset/modelo (`best_car_model_sprint3.pkl`, `best_car_model.pkl`).
- `sprint2_preparacion.ipynb`, `sprint3_modelamiento.ipynb` — notebooks de preparación y modelado.
- `requirements.txt` — dependencias de Python.
- `render.yaml` — configuración de despliegue para Render.

## Ejecutar local
```bash
pip install -r requirements.txt
python app.py
```
Abre `http://localhost:5000` para usar la UI o envía un POST a `/predict`.

### Ejemplo de petición
```json
{
  "buying": "low",
  "maint": "med",
  "doors": "4",
  "persons": "4",
  "lug_boot": "big",
  "safety": "high"
}
```
Respuesta:
```json
{
  "prediction": "Aceptable",
  "prediction_code": "acc"
}
```

## Despliegue (Render)
Render ejecuta `pip install -r requirements.txt` y arranca con `gunicorn app:app` según `render.yaml`. Ajusta variables o nombre del servicio ahí si lo necesitas.***
