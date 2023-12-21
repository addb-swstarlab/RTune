#!/usr/bin/env bash

# Use nohup to run even if the terminal is closed, and use & to run the command in background
nohup sudo python server.py >> server.log &
