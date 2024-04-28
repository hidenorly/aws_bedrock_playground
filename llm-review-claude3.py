#   Copyright 2024 hidenorly
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import argparse
import os
import sys
import json
import logging
import boto3
from botocore.exceptions import ClientError

def files_reader(files):
    result = ""

    for path in files:
        if os.path.exists( path ):
          with open(path, 'r', encoding='UTF-8') as f:
            result += f.read()
            f.close()

    return result

def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens):
    """
    Streams the response from a multimodal prompt.
    Args:
        bedrock_runtime: The Amazon Bedrock boto3 client.
        model_id (str): The model ID to use.
        system_prompt (str) : The system prompt text.
        messages (JSON) : The messages to send to the model.
        max_tokens (int) : The maximum  number of tokens to generate.
    """

    result = ""
    status = {}

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 1,  # default value is 1
            "top_p": 0.999,    # default value is 0.999
            "system": system_prompt,
            "messages": messages
        }
    )

    response = bedrock_runtime.invoke_model_with_response_stream(
        body=body,
        modelId=model_id)

    for event in response.get("body"):
        chunk = json.loads(event["chunk"]["bytes"])

        if chunk['type'] == 'message_delta':
            status = {
                "stop_reason" : chunk['delta']['stop_reason'],
                "stop_sequence" : chunk['delta']['stop_sequence'],
                "output_tokens" : chunk['usage']['output_tokens'],
            }
        if chunk['type'] == 'content_block_delta':
            if chunk['delta']['type'] == 'text_delta':
                result += chunk['delta']['text']

    return result, status


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Code review specified file with OpenAI LLM')
    parser.add_argument('args', nargs='*', help='files')
    parser.add_argument('-k', '--accessKey', action='store', default=os.getenv('AWS_ACCESS_KEY_ID'), help='specify your access key or set in .aws/credential or set in AWS_ACCESS_KEY_ID env')
    parser.add_argument('-s', '--secretKey', action='store', default=os.getenv('AWS_SECRET_ACCESS_KEY'), help='specify your secret key or set it .aws/credential or set in AWS_SECRET_ACCESS_KEY')
    parser.add_argument('-r', '--region', action='store', default="us-west-2", help='specify region')
    parser.add_argument('-m', '--model', action='store', default="anthropic.claude-3-sonnet-20240229-v1:0", help='specify model')
    args = parser.parse_args()

    codes = ""
    if len(args.args)>0:
        codes = files_reader(args.args)
    else:
        codes = sys.stdin.read()

    system_prompt = "You're the world class best programmer and you're doing pair programming. You're requeted to code-review. You need pointed out what's problem, the potential risk and the future expansion. And you need to explain how to solve with expected examples. Example code is expected as diff output manner as -:original code +:modified code"
    user_prompt = "Please review the following code and please explain the problem and please show the better code about the problematic part.\n"+codes

    logger = logging.getLogger(__name__)
    #logging.basicConfig(level=logging.INFO)
    status = None

    try:
        bedrock_runtime = None
        if args.accessKey and args.secretKey and args.region:
            bedrock_runtime = boto3.client(
                service_name = 'bedrock-runtime',
                aws_access_key_id = args.accessKey,
                aws_secret_access_key = args.secretKey,
                region_name = args.region
            )
        else:
            # Use .aws/credential's default
            bedrock_runtime = boto3.client(service_name='bedrock-runtime')

        if bedrock_runtime:
            model_id = args.model

            max_tokens = 50000
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
            messages = [user_message]

            response_messages, status = generate_message(
                bedrock_runtime, model_id, system_prompt, messages, max_tokens)

            print( response_messages )
            #print(str(status))

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        print(str(status))
