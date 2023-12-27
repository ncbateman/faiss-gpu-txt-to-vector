# Build Docker images
docker build -f ./docker/base/Dockerfile -t faiss-gpu-txt-to-vector:latest .
docker stop faiss-gpu-txt-to-vector-test
docker rm faiss-gpu-txt-to-vector-test
docker build -f ./docker/test/Dockerfile -t faiss-gpu-txt-to-vector-test:latest .
docker run --name faiss-gpu-txt-to-vector-test faiss-gpu-txt-to-vector-test:latest