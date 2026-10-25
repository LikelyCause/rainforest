"""The seeder command Compose runs (`python -m app.sample_mongo`)."""

import os

import pymongo
import pytest

from app import sample_mongo


@pytest.fixture
def db(mongo_url):
    client = pymongo.MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    database = client[f"rf_test_cli_{os.getpid()}"]
    client.drop_database(database.name)
    yield database
    client.drop_database(database.name)
    client.close()


def test_main_seeds_both_collections(mongo_url, db):
    sample_mongo.main([mongo_url, db.name])
    assert db["curated"].count_documents({}) == sample_mongo.CURATED_DOCS
    assert db["massive"].count_documents({}) == sample_mongo.MASSIVE_DOCS


def test_main_replaces_what_is_there_without_if_empty(mongo_url, db):
    sample_mongo.main([mongo_url, db.name])
    db["curated"].delete_many({})
    sample_mongo.main([mongo_url, db.name])
    assert db["curated"].count_documents({}) == sample_mongo.CURATED_DOCS


def test_if_empty_seeds_an_empty_database(mongo_url, db):
    sample_mongo.main([mongo_url, db.name, "--if-empty"])
    assert db["curated"].count_documents({}) == sample_mongo.CURATED_DOCS


def test_if_empty_leaves_a_database_with_collections_alone(mongo_url, db):
    sample_mongo.main([mongo_url, db.name])
    db["curated"].delete_many({})
    db["curated"].insert_one({"edited": True})
    sample_mongo.main([mongo_url, db.name, "--if-empty"])
    assert db["curated"].count_documents({}) == 1
