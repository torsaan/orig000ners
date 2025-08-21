import requests
import json

def get_all_nvdb_object_types():
    """
    Fetches all object types from the Norwegian National Road Database (NVDB) API.

    Returns:
        A list of dictionaries, where each dictionary represents an object type.
        Returns None if the request fails.
    """
    # The base URL for the NVDB API v3
    api_url = "https://nvdbapiles-v3.atlas.vegvesen.no/vegobjekttyper"
    
    # Set headers to specify that we want the response in JSON format
    headers = {
        'Accept': 'application/vnd.vegvesen.nvdb-v3-rev1+json'
    }

    try:
        print("Fetching data from the NVDB API...")
        # Make the GET request to the API
        response = requests.get(api_url, headers=headers)
        
        # Raise an exception if the request was unsuccessful (e.g., 404, 500)
        response.raise_for_status()
        
        print("Successfully retrieved data!")
        # Parse the JSON response and return it
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while communicating with the API: {e}")
        return None

if __name__ == "__main__":
    object_types = get_all_nvdb_object_types()
    
    if object_types:
        print(f"\nFound a total of {len(object_types)} NVDB object types.")
        print("-" * 30)
        
        # Loop through the list and print the ID and name of each object type
        for obj_type in object_types:
            # The 'id' and 'navn' keys hold the object's ID and name respectively
            obj_id = obj_type.get('id')
            obj_name = obj_type.get('navn')
            print(f"ID: {obj_id:<5} | Name: {obj_name}")
            
        # Example of how to save the full data to a file
        try:
            with open("nvdb_object_types.json", "w", encoding="utf-8") as f:
                json.dump(object_types, f, ensure_ascii=False, indent=4)
            print("\n✅ Successfully saved the complete list to 'nvdb_object_types.json'")
        except IOError as e:
            print(f"\nCould not write to file: {e}")