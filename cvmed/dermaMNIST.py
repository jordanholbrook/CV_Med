# Databricks notebook source
from medmnist import DermaMNIST
train_dataset = DermaMNIST(split="train", download=True)
test_dataset = DermaMNIST(split="test", download=True)
test_dataset.montage(length=30)

# COMMAND ----------


