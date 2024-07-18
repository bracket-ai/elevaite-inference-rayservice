# ElevAIte Inference Rayservice

This is the "`working_dir`" (i.e. functional source code) for the ElevAIte inference RayService.

## Installation

In your preferred Python environment:

```shell
pip install -r requirements.txt
```

## Run (development environment)

To run the dev server:

```shell
serve run endpoint:deployment
```

To test the server (assuming it is run with the default NER model):

```shell
curl -X 'POST'   'http://localhost:8000/infer'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{"args": ["Hello, world, this is John Smith"], "kwargs": {}}'
```



