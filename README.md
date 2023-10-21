# ai-server

weights file: https://drive.google.com/file/d/1VH0oJikm3Z9k48q59SayyhMlK-nj5PsC/view?usp=sharing

## Test Locally

1. First link the weights file:
```
ln -s /path/to/weights.pt ./model.pt
```
2. Install requirements
```
pip install -r requirements.txt
```
3. Run the server locally with:
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
