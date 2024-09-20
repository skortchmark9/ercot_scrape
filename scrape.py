import requests
import time
import os
import zipfile
import tempfile

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


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
        self.request_params = {
            'postDatetimeFrom': '2019-01-01T00:00',
            'postDatetimeTo': '2024-01-01T00:00',
            'size': 1000,  # Get all links in one page
        }
        # Initialize session
        self.session = requests.Session()
        self.session.hooks['response'] = [self._handle_401]

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

    def index_request(self, url):
        all_matches = []
        for i in range(1, 3):  # Two pages of requests
            response = self.session.get(url, params=dict(**self.request_params, page=i))
            response.raise_for_status()
            data = response.json()
            archives = data.get('archives', [])
            all_matches += [{
                'name': a['friendlyName'],
                'link': a['_links']['endpoint']['href'],
                'id': a['docId'],
            } for a in archives]
        return all_matches

    def download_files(self, files, folder):
        for i, file in enumerate(files):
            print(f'downloading {i + 1}/{len(files)}')
            time.sleep(2)
            response = self.session.get(file['link'])
            response.raise_for_status()
            filename = file['link'].split('?')[1].split('=')[1]
            with open(f'{folder}/{filename}.csv', "w") as f:
                f.write(response.text)
    
    def download_dam_shadow_files(self):
        url = f'{self.base_url}/{self.dam_shadow_prices}'
        all_files = self.index_request(url)
        self.download_files(all_files, 'dam_shadow')
    
    def download_dam_lmps_files(self):
        url = f'{self.base_url}/{self.dam_hourly_lmps}'
        all_files = self.index_request(url)
        self.download_files(all_files, 'dam_lmps')
    
    def download_ldf_files(self):
        url = f'{self.base_url}/{self.ldf}'
        all_files = self.index_request(url)
        self.download_files(all_files, 'ldf')

    def ldf_download(self):
        ldf_link = 'https://api.ercot.com/api/public-reports/archive/np4-159-cd?download=1023584516'
        response = self.session.get(ldf_link)
        response.raise_for_status()
        return response.text

    def batch_download(self, archive_type, batch_size=500):
        INDEX_URL = 'https://api.ercot.com/api/public-reports/archive'
        url = INDEX_URL + '/' + archive_type
        all_files = self.index_request(url)
        dest_dir = archive_type

        print(f'downloading {len(all_files)} files to {dest_dir}...')


        os.makedirs(dest_dir, exist_ok=True)

        batches = batch(all_files, batch_size)
        print(f'{len(batches)} of {batch_size} files')

        with tempfile.TemporaryDirectory() as tmpdir:
            for batch_no, file_batch in enumerate(batches):
                print(f'downloading batch {batch_no} / {len(batches)}')
                batch_no += 1
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
                print(f'batch {batch_no} downloaded in {end_dl - start_dl}s, unzipped in {end_extract - end_dl}')

        for zipped_csv in os.listdir(dest_dir):
            if not zipped_csv.endswith('.zip'):
                continue
            path = dest_dir + "/" + zipped_csv
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dest_dir)
            os.remove(path)


            

        

# Example usage
***REMOVED***
