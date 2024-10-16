from dataclasses import dataclass
import random
import requests
import time
import os
import zipfile
import tempfile
import pandas as pd
import glob
import urllib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

@dataclass
class Archive:
    base_url = 'https://api.ercot.com/api/public-reports/archive'
    id: str;
    folder: str;
    batch_size = 500

    @property
    def url(self):
        return self.base_url + '/' + self.id


LDF = Archive('NP4-159-CD', 'ldf')
DamShadow = Archive('NP4-191-CD', 'dam_shadow')
DamHourlyLmps = Archive('NP4-183-CD', 'dam_hourly')
WeatherZoneLoadForecast7D = Archive('NP3-565-CD', 'weatherzone_load_forecast')
ActualWind = Archive('NP4-733-CD', 'wind')
ActualWind.batch_size = 1000
ActualSolar = Archive('NP4-738-CD', 'solar')
ActualSolar.batch_size = 1000



class ErcotApiClient:
    def __init__(self, username, password, subscription_key):
        self.username = username
        self.password = password
        self.subscription_key = subscription_key
        self.access_token = None
        self.refresh_token = None
        self.base_url = 'https://api.ercot.com/api/public-reports/archive'
        self.dam_shadow_prices = 'NP4-191-CD'
        self.ldf = 'NP4-159-CD'
        self.dam_hourly_lmps = 'NP4-183-CD'
        self.executor = None
        self.request_params = {
            'postDatetimeFrom': '2019-01-01T00:00',
            'postDatetimeTo': '2024-01-01T00:00',
            'size': 1000,  # Get all links in one page
        }

        # Initialize session
        self.session = requests.Session()
        self.session.hooks['response'] = [self._handle_401, self._handle_429]

    def authenticate(self):
        auth_url = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
        data = {
            'username': self.username,
            'password': self.password,
            'grant_type': 'password',
            'response_type': 'id_token',
            'scope': 'openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access',
            'client_id': 'fec253ea-0d06-4272-a5e6-b478baeecd70',
        }
        response = self.session.post(auth_url, data=data)
        response.raise_for_status()
        token_data = response.json()
        self.access_token = token_data['access_token']
        self.refresh_token = token_data['refresh_token']
        self._update_headers()

    def refresh_access_token(self):
        auth_url = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
        data = {    
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'response_type': 'id_token',
            'scope': 'openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access',
            'client_id': 'fec253ea-0d06-4272-a5e6-b478baeecd70',
        }
        response = self.session.post(auth_url, data=data)
        response.raise_for_status()
        token_data = response.json()
        self.access_token = token_data['access_token']
        self.refresh_token = token_data['refresh_token']
        self._update_headers()

    def _update_headers(self):
        """Update session headers with the latest token."""
        self.session.headers.update({
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Authorization': f'Bearer {self.access_token}'
        })

    def _handle_401(self, response, *args, **kwargs):
        """Handle 401 responses by refreshing the token and retrying the request."""
        if response.status_code == 401:
            print("401 Unauthorized - refreshing token and retrying...")
            self.refresh_access_token()
            # Retry the request with the refreshed token
            request = response.request
            request.headers['Authorization'] = f'Bearer {self.access_token}'
            return self.session.send(request)
        return response
    
    def _handle_429(self, response, *args, **kwargs):
        """Handle 429 responses by refreshing the token and retrying the request."""
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After', 5)
            if retry_after:
                print(f"429 Rate limited. Waiting {retry_after}s and trying again")
            time.sleep(int(retry_after) + random.randint(0, 20) * 0.1)
            request = response.request
            return self.session.send(request)
        return response

    def get_from_cache(self, prepared_request):
        path = 'request_cache'
        cache_key = prepared_request.url
        encoded = urllib.parse.quote_plus(cache_key)
        response_data = None
        if os.path.exists(path + '/' + encoded):
            with open(path + '/' + encoded) as f:
                contents = f.read()
                # Race condition where file was written without
                # any data being written
                if not contents:
                    return None
                
                response_data = json.loads(contents)
                
                return response_data

    def write_to_cache(self, prepared_request, data):
        path = 'request_cache'
        os.makedirs(path, exist_ok=True)
        cache_key = prepared_request.url
        encoded = urllib.parse.quote_plus(cache_key)
        with open(path + '/' + encoded, 'w+') as f:
            f.write(json.dumps(data))

    def index_request(self, archive):
        all_matches = []
        for i in range(1, 1000):  # Two pages of requests
            prepared_request = requests.Request('GET', archive.url, params=dict(**self.request_params, page=i)).prepare()
            data = self.get_from_cache(prepared_request)
            if not data:
                response = self.session.get(archive.url, params=dict(**self.request_params, page=i))
                if i % 10 == 0:
                    print(f'Requesting page {i}')
                # Probably because we're out of pages
                if response.status_code == 400:
                    break
                response.raise_for_status()
                data = response.json()
                self.write_to_cache(prepared_request, data)

            archives = data.get('archives', [])
            all_matches += [{
                'name': a['friendlyName'],
                'link': a['_links']['endpoint']['href'],
                'id': a['docId'],
            } for a in archives]
        return all_matches

    def batch_download(self, archive):
        url = archive.url
        all_files = self.index_request(archive)
        dest_dir = archive.folder

        print(f'downloading {len(all_files)} files to {dest_dir}...')

        os.makedirs(dest_dir, exist_ok=True)

        batches = list(batch(all_files, archive.batch_size))
        print(f'{len(batches)} batches of {archive.batch_size} files')

        with tempfile.TemporaryDirectory() as tmpdir:
            for batch_no, file_batch in enumerate(batches):
                print(f'downloading batch {batch_no + 1} / {len(batches)}')
                docIds = [str(doc['id']) for doc in file_batch]
                data = {
                    'docIds': docIds,
                }
                start_dl = time.time()
                response = self.session.post(url + '/download', 
                    json=data,
                    stream=True
                )
                response.raise_for_status()
                path = f"{tmpdir}/batch_{batch_no}.zip"
                with open(path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)

                end_dl = time.time()
                with zipfile.ZipFile(path) as zf:
                    zf.extractall(dest_dir)

                end_extract = time.time()
                print(f'batch {batch_no} downloaded in {end_dl - start_dl}s, unzipped in {end_extract - end_dl}s')

        for zipped_csv in os.listdir(dest_dir):
            if not zipped_csv.endswith('.zip'):
                continue
            path = dest_dir + "/" + zipped_csv
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dest_dir)
            os.remove(path)

    def batch_download_parallel(self, archive):
        url = archive.url
        all_files = self.index_request(archive)
        dest_dir = archive.folder

        print(f'downloading {len(all_files)} files to {dest_dir}...')

        os.makedirs(dest_dir, exist_ok=True)

        batches = list(batch(all_files, archive.batch_size))
        processed = 0
        print(f'{len(batches)} batches of {archive.batch_size} files')
        overall_start = time.time()

        def download_and_unzip_batch(batch_no, file_batch, tmpdir):
            docIds = [str(doc['id']) for doc in file_batch]
            data = {
                'docIds': docIds,
            }
            start_dl = time.time()
            response = self.session.post(url + '/download', 
                                        json=data, 
                                        stream=True)
            response.raise_for_status()
            path = f"{tmpdir}/batch_{batch_no}.zip"
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

            end_dl = time.time()
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dest_dir)

            end_extract = time.time()
            print(f'batch {batch_no} downloaded in {end_dl - start_dl}s, unzipped in {end_extract - end_dl}s')

        with tempfile.TemporaryDirectory() as tmpdir:
            with ThreadPoolExecutor(max_workers=10) as executor:
                self.executor = executor
                futures = [
                    executor.submit(download_and_unzip_batch, batch_no, file_batch, tmpdir)
                    for batch_no, file_batch in enumerate(batches)
                ]
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f'Error occurred: {e}')

                    processed += 1
                    print(f"{processed}/{len(batches)} done")

        # Unzipping any additional files that were zipped in the destination directory
        beginning_unzipping = time.time()
        print(f"Downloads complete in {beginning_unzipping - overall_start}s...unzipping downloaded files")
        for zipped_csv in os.listdir(dest_dir):
            if not zipped_csv.endswith('.zip'):
                continue
            path = dest_dir + "/" + zipped_csv
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dest_dir)
            os.remove(path)

        done = time.time()
        print(f"Unzipped in {done - beginning_unzipping}s")
        print(f"Downloaded {len(all_files)} in {done - overall_start}s.")


def merge_csvs(dir):
    # Use glob to find all CSV files in the directory and sort them
    start = time.time()
    csv_files = sorted(glob.glob(f'{dir}/*.csv'))

    # Create an empty list to store dataframes
    dfs = []

    # Loop over all CSV files and read them into pandas dataframes
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    t1 = time.time()
    print(f"Read all csvs in {t1 - start}")
    # Concatenate all dataframes into one dataframe
    merged_df = pd.concat(dfs, ignore_index=True)
    t2 = time.time()
    print(f"Concatenated in in {t2 - t1}")

    return merged_df

    merged_df.to_csv(dir + '.csv')
    t3 = time.time()
    print(f"Wrote to disk in {t3 - t2}")
        

# Example usage
***REMOVED***
