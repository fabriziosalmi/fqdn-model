# Usage

## CLI Prediction

```bash
python predict.py google.com
```

## API Server

Start the server:
```bash
python api.py
```

Query the API:
```bash
curl -X POST http://localhost:5000/predict -d '{"fqdn":"example.com"}' -H "Content-Type: application/json"
```
