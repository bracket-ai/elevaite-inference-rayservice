# ElevAIte Inference Rayservice

This is the "`working_dir`" (i.e. functional source code) for the ElevAIte inference RayService.

## Installation

In your preferred Python environment:

```shell
pip install -r requirements.txt
```

## Run (development environment)

NOTE: This RayService is only designed to run Huggingface models using the `transformers` library

To run the dev server, make sure that an FTP server is available and serving the desired model. Pass that model's URL in
a format compatible with [`elevaite-file-client`](https://github.com/bracket-ai/elevaite-file-client) via
the `MODEL_URL` environment variable and the
pipeline [task](https://github.com/bracket-ai/elevaite/blob/abcb7bc29c36113809f8e0b1dc68fcb642c31c5d/elevaite_worker/models.py#L7)
via the `TASK` environment variable


```shell
MODEL_URL="ftp://username:password@localhost/ftp/default/models/1" TASK="token-classification" serve run endpoint:deployment
```

*The above example assumes that a model of the `token-classification` task is being served from the
directory `/ftp/default/models/1` on an FTP server on localhost (readable to username `username` and
password `password`). It is your responsibility to make sure that is true!*

To test the server (assuming it is run with the default NER model):

```shell
curl -X 'POST'   'http://localhost:8000/infer'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{"args": ["Hello, world, this is John Smith"], "kwargs": {}}'
```



