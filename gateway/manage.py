#!/usr/bin/env python3
"""Create gateway API keys and register/unregister inference workers."""
from __future__ import annotations

import argparse
import hashlib
import secrets

import boto3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="personal")
    parser.add_argument("--region", default="us-west-2")
    sub = parser.add_subparsers(dest="command", required=True)

    key = sub.add_parser("create-key")
    key.add_argument("--table", required=True)
    key.add_argument("--customer", required=True)

    register = sub.add_parser("register-worker")
    register.add_argument("--service-id", required=True)
    register.add_argument("--instance-id", required=True)
    register.add_argument("--url", required=True)

    unregister = sub.add_parser("unregister-worker")
    unregister.add_argument("--service-id", required=True)
    unregister.add_argument("--instance-id", required=True)
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    if args.command == "create-key":
        value = "rdb_" + secrets.token_urlsafe(32)
        session.client("dynamodb").put_item(
            TableName=args.table,
            Item={
                "key_hash": {"S": hashlib.sha256(value.encode()).hexdigest()},
                "customer_id": {"S": args.customer},
                "enabled": {"BOOL": True},
            },
            ConditionExpression="attribute_not_exists(key_hash)",
        )
        print(value)  # Shown once; only the SHA-256 digest is stored.
    elif args.command == "register-worker":
        session.client("servicediscovery").register_instance(
            ServiceId=args.service_id,
            InstanceId=args.instance_id,
            Attributes={
                "INSTANCE_URL": args.url.rstrip("/"),
                "AWS_INIT_HEALTH_STATUS": "HEALTHY",
            },
        )
    else:
        session.client("servicediscovery").deregister_instance(
            ServiceId=args.service_id, InstanceId=args.instance_id)


if __name__ == "__main__":
    main()
