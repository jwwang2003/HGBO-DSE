# HGBO-DSE Backend Development Notes

## Running the backend of HGBO-DSE

Start the Docker container for running the:
```
docker run --rm -it -v /home/wjw/tools/xilinx:/home/wjw/tools/xilinx:ro -p 8000:8000 <image-name>
```

### Bulding the Docker container for Celery Backend

Run the following command in the root of the project directory to build the container:
```
docker build -t <image-name> .
```
