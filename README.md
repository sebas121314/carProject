# Car Evaluation - Despliegue (Sprint 4)

App Flask que sirve el modelo entrenado del dataset Car Evaluation (UCI) y expone un endpoint de predicción con UI web.

## Estructura
- `app.py`: API + UI.
- `templates/index.html`: formulario web.
- `static/styles.css`: estilos.
- `car+evaluation/`: datos y modelo (`best_car_model_sprint3.pkl` preferido; `best_car_model.pkl` como respaldo).
- `sprint2_preparacion.ipynb`, `sprint3_modelamiento.ipynb`: notebooks de preparación y modelamiento.
- `requirements.txt`: dependencias.

## Ejecutar local
```bash
pip install -r requirements.txt
python app.py
```
Abre `http://localhost:5000` y usa el formulario o prueba con Thunder Client / Postman.

## API
- `GET /` → UI.
- `POST /predict` → JSON o form-data con las 6 features:
  - `buying`: vhigh|high|med|low
  - `maint`: vhigh|high|med|low
  - `doors`: 2|3|4|5more
  - `persons`: 2|4|more
  - `lug_boot`: small|med|big
  - `safety`: low|med|high

Ejemplo body:
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

## Notas
- El modelo se carga desde `car+evaluation/`; si falta, re-ejecuta `sprint3_modelamiento.ipynb` para regenerarlo.
- Se eliminó el modelo duplicado en la raíz; mantén los `.pkl` dentro de `car+evaluation/`.
