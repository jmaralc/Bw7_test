IMAGE_NAME=bw7-jma

build:
	docker build -q -t $(IMAGE_NAME) .

run: build
	 docker run -p 8888:8888 $(IMAGE_NAME)

delete:
	docker rm $(IMAGE_NAME)