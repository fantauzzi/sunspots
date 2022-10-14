#!/bin/bash

cd server || exit
PYTHONPATH='..' uvicorn server:app --reload
