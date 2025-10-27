import argparse
import requests
import google.auth
import google.auth.transport.requests

def trigger_cloud_build(project_id: str, trigger_id: str, region: str = "europe-west4", branch: str = "frontend"):
    print(f"Triggering Cloud Build job '{trigger_id}' in project '{project_id}' (region: {region}, branch: {branch}).")
    print("Acquiring authentication credentials.")
    creds, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    access_token = creds.token
    
    url = f"https://cloudbuild.googleapis.com/v1/projects/{project_id}/locations/{region}/triggers/{trigger_id}:run"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "projectId": project_id,
        "triggerId": trigger_id,
        "source": {
            "projectId": project_id,
            "branchName": branch
        }
    }
    
    print(f"Sending request to Cloud Build API: {url}")
    print(f"Request payload: {data}")
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print("Cloud Build trigger was successfully initiated.")
        print(f"API Response: {response.json()}")
    else:
        print(f"Error triggering Cloud Build. Status code: {response.status_code}")
        print(f"Response content: {response.text}")
        raise RuntimeError("Failed to trigger the deployment pipeline.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=str, required=True)
    parser.add_argument("--trigger-id", type=str, required=True)
    parser.add_argument("--region", type=str, default="europe-west4")
    parser.add_argument("--branch", type=str, default="frontend")
    args = parser.parse_args()
    trigger_cloud_build(args.project_id, args.trigger_id, args.region, args.branch)