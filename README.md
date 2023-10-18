# ai-server

## Test Locally

1. First link the weights file:
```
ln -s /path/to/weights.pt ./model.pt
```

2. Run the server locally with:
```
python app.py
```

## Docker
1. Copy the weights file to the root directory.
```
cp /path/to/weights.pt ./model.pt
```
2. Containerize the server with:
```
docker build -t ai-server .
```
3. Run the container with:
```
docker run -p 127.0.0.1:5000:80 ai-server
```
