import argparse
import requests
import google.auth
import google.auth.transport.requests

def trigger_cloud_build(project_id: str, trigger_id: str, new_model_uri: str, best_model_name: str):
    """Triggers a Cloud Build pipeline and passes substitution variables."""
    
    print(f"Attempting to trigger Cloud Build: project='{project_id}', trigger='{trigger_id}'")
    
    creds, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    access_token = creds.token

    url = f"https://cloudbuild.googleapis.com/v1/projects/{project_id}/triggers/{trigger_id}:run"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    data = {
        "substitutions": {
            "_NEW_MODEL_URI": new_model_uri,
            "_BEST_MODEL_NAME": best_model_name
        }
    }
    print(f"Payload: {data}")

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        print("Cloud Build trigger successful.")
        print(f"Response: {response.json()}")
    else:
        print(f"Error triggering Cloud Build. Status: {response.status_code}")
        print(f"Response text: {response.text}")
        raise RuntimeError("Failed to trigger Cloud Build pipeline.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', type=str, required=True)
    parser.add_argument('--trigger-id', type=str, required=True)
    parser.add_argument('--new-model-uri', type=str, required=True)
    parser.add_argument('--best-model-name', type=str, required=True)
    args = parser.parse_args()
    trigger_cloud_build(args.project_id, args.trigger_id, args.new_model_uri, args.best_model_name)
