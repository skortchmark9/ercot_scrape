import requests
import time


# https://api.ercot.com/api/public-reports/archive/np4-191-cd/download
x = { "docIds": [
    "1037369000",
    "1037095438",
    "1036825492",
    "1036548570",
    "1036279201",
    "1036020924",
    "1035755848",
    "1035487323",
    "1035218453",
    "1034946124",
    "1034670022",
    "1034402306",
    "1034145937",
    "1033883200",
    "1033608445",
    "1033337476",
    "1033061818",
    "1032792879",
    "1032535528",
    "1032279053",
    "1032019067",
    "1031753569",
    "1031489461",
    "1031223381",
    "1030946090",
    "1030678633",
    "1030422344",
    "1030163676",
    "1029897460",
    "1029630949",
    "1029363741",
    "1029088081",
    "1028820653",
    "1028564179",
    "1028304225",
    "1028037077",
    "1027771841",
    "1027508252",
    "1027237807",
    "1026973807",
    "1026720855",
    "1026465331",
    "1026200670",
    "1025937108",
    "1025673013",
    "1025398129",
    "1025134193",
    "1024880487",
    "1024622864",
    "1024359061",
    "1024098288",
    "1023833936",
    "1023561789",
    "1023297671",
    "1023046578",
    "1022790156",
    "1022523424",
    "1022261847",
    "1021998559",
    "1021726945",
    "1021463021",
    "1021211497",
    "1020960963",
    "1020693839",
    "1020431014",
    "1020167966",
    "1019897330",
    "1019634237",
    "1019381524",
    "1019125188",
    "1018861964",
    "1018599495",
    "1018334819",
    "1018064181",
    "1017799689",
    "1017547170",
    "1017284198",
    "1017021593",
    "1016762195",
    "1016507085",
    "1016235647",
    "1015972615",
    "1015719241",
    "1015464075",
    "1015200663",
    "1014937535",
    "1014675316",
    "1014401178",
    "1014136194",
    "1013883640",
    "1013628417",
    "1013365958",
    "1013105764",
    "1012848394",
    "1012573233",
    "1012317292",
    "1012066205",
    "1011810648",
    "1011549443",
    "1011289397",
    "1011027687",
    "1010755743",
    "1010493525",
    "1010242535",
    "1009989224",
    "1009727267",
    "1009465207",
    "1009205390",
    "1008936575",
    "1008675540",
    "1008425728",
    "1008173789",
    "1007915059",
    "1007656573",
    "1007375837",
    "1007123611",
    "1006871839",
    "1006623069",
    "1006376462",
    "1006105853",
    "1005846674",
    "1005593647",
    "1005316192",
    "1005053638",
    "1004801593",
    "1004549735",
    "1004293046",
    "1004034376",
    "1003770219",
    "1003499057",
    "1003235472",
    "1002984559",
    "1002728229",
    "1002468543",
    "1002203307",
    "1001939835",
    "1001665176",
    "1001403241",
    "1001151545",
    "1000898341",
    "1000635663",
    "1000377192",
    "1000107337",
    "999849793",
    "999579858",
    "999330894",
    "999077782",
    "998823653",
    "998566385",
    "998307124",
    "998045328",
    "997787931",
    "997540343",
    "997289555",
    "997030047",
    "996780928",
    "996520566",
    "996258809",
    "995994915",
    "995748883",
    "995497473",
    "995244694",
    "994993258",
    "994735337",
    "994474167",
    "994215315",
    "993966218",
    "993720794",
    "993465278",
    "993211721",
    "992959016",
    "992694794",
    "992442637",
    "992191878",
    "991946161",
    "991695840",
    "991438598",
    "991177893",
    "990917369",
    "990661177",
    "990419285",
    "990170907",
    "989915418",
    "989659277",
    "989411949",
    "989144218",
    "988884451",
    "988640672",
    "988394868",
    "988141968",
    "987888404",
    "987639922",
    "987375506",
    "987121147",
    "986886849",
    "986642892",
    "986394133",
    "986137452",
    "985876485",
    "985612475",
    "985357168",
    "985115969",
    "984864777",
    "984613454",
    "984360028",
    "984115463",
    "983845866",
    "983597357",
    "983354446",
    "983109051",
    "982857894",
    "982611518",
    "982361960",
    "982100442",
    "981850821",
    "981609646",
    "981383660",
    "981120631",
    "980872472",
    "980623025",
    "980361580",
    "980109414",
    "979870853",
    "979629304",
    "979379189",
    "979130091",
    "978883008",
    "978616220",
    "978365530",
    "978124464",
    "977882068",
    "977632233",
    "977382303",
    "977126767",
    "976866045",
    "976613558",
    "976370186",
    "976130888",
    "975875045",
    "975620200",
    "975362114",
    "975107023",
    "974854233",
    "974613582",
    "974370346",
    "974122046",
    "973869681",
    "973606654",
    "973346636",
    "973100165",
    "972856883",
    "972608618",
    "972356567",
    "972108506",
    "971863925",
    "971597062",
    "971345355",
    "971103756",
    "970861600",
    "970610088",
    "970358427",
    "970097614",
    "969849916",
    "969612577",
    "969376323",
    "969137288",
    "968885705",
    "968624450",
    "968374552",
    "968134685",
    "967898147",
    "967661531",
    "967422614",
    "967174510",
    "966929013",
    "966681544",
    "966427628",
    "966177896",
    "965939007",
    "965699935",
    "965448241",
    "965199539",
    "964958769",
    "964704597",
    "964455435",
    "964224499",
    "963990290",
    "963747343",
    "963506532",
    "963265910",
    "963006977",
    "962764795",
    "962536903",
    "962296559",
    "962058443",
    "961815383",
    "961566371",
    "961306428",
    "961062031",
    "960830952",
    "960600691",
    "960368913",
    "960136114",
    "959891214",
    "959634956",
    "959401613",
    "959169554",
    "958935759",
    "958694038",
    "958453183",
    "958213298",
    "957963708",
    "957724096",
    "957491339",
    "957259239",
    "957017157",
    "956780467",
    "956542033",
    "956288633",
    "956058695",
    "955821901",
    "955591369",
    "955353593",
    "955118025",
    "954878508",
    "954637495",
    "954391012",
    "954161130",
    "953924499",
    "953681666",
    "953438184",
    "953202061",
    "952949452",
    "952755210",
    "952482897",
    "952253709",
    "952015819",
    "951783995",
    "951552961",
    "951306766",
    "951072467",
    "950847310",
    "950618732",
    "950383296",
    "950151505",
    "949917333",
    "949675382",
    "949442492",
    "949218495",
    "948990147",
    "948753617",
    "948517870",
    "948281160",
    "948040399",
    "947804462",
    "947581691",
    "947351406",
    "947118957",
    "946879181",
    "946640226",
    "946392653",
    "946154747",
    "945926974",
    "945695855",
    "945459896",
    "945224581",
    "944987549",
    "944742041",
    "944510749",
    "944280088",
    "944050265",
    "943815299",
    "943579704",
    "943343526",
    "943100031",
    "942862500",
    "942634234",
    "942402596",
    "942164954",
    "941916080",
    "941670781",
    "941433920",
    "941207153",
    "940975630",
    "940744792",
    "940505566",
    "940269413",
    "940033526",
    "939788300",
    "939549925",
    "939321820",
    "939089445",
    "938848677",
    "938612253",
    "938374667",
    "938130496",
    "937900884",
    "937666253",
    "937436889",
    "937201010",
    "936966054",
    "936729715",
    "936486057",
    "936250266",
    "936022916",
    "935792505",
    "935554512",
    "935317694",
    "935083086",
    "934837535",
    "934603016",
    "934378620",
    "934152105",
    "933915421",
    "933681542",
    "933448271",
    "933208580",
    "932974406",
    "932750434",
    "932522923",
    "932288341",
    "932052114",
    "931816974",
    "931573832",
    "931338886",
    "931115375",
    "930887793",
    "930652271",
    "930418497",
    "930189936",
    "929953828",
    "929723831",
    "929503724",
    "929273325",
    "929048380",
    "928816411",
    "928592224",
    "928351950",
    "928116633",
    "927895319",
    "927670222",
    "927439682",
    "927208604",
    "926984187",
    "926755951",
    "926527994",
    "926309320",
    "926089079",
    "925861070",
    "925636977",
    "925408072",
    "925173532",
    "924937341",
    "924717295",
    "924496798",
    "924267165",
    "924040317",
    "923809840",
    "923571488",
    "923341871",
    "923121931",
    "922905111",
    "922671748",
    "922442280",
    "922212010",
    "921983980",
    "921750104",
    "921530333",
    "921316172",
    "921078931",
    "920850993",
    "920621577",
    "920383007",
    "920154515",
    "919935298",
    "919712455",
    "919485420",
    "919254770",
    "919019884",
    "918790458",
    "918573783",
    "918357160",
    "918136291",
    "917908012",
    "917682810",
    "917455581",
    "917222755",
    "916996453",
    "916781002",
    "916562416",
    "916335649",
    "916110100",
    "915882211",
    "915646146",
    "915419262",
    "915199936",
    "914983409",
    "914762626",
    "914534775",
    "914311007"
  ]
}


# LOAD DISTRIBUTION FACTORS
"""
{
    '_meta': {'totalRecords': 49808, 'pageSize': 10, 'totalPages': 4981, 'currentPage': 1,
    'query': {'parameterCount': 0, 'parameters': {}, 'sortedBy': 'postDatetime: DESC'}},
    'product': {'emilId': 'NP4-159-CD', 'name': 'Load Distribution Factors', reportTypeId': 12324 },
    'archives': [{
        'docId': 1025479422, 'friendlyName': 'LDFNP4159_csv', 'postDatetime': '2024-08-05T19:40:39.000',
        '_links': {'endpoint': {'href': 'https://api.ercot.com/api/public-reports/archive/np4-159-cd?download=1025479422'}}
    },
      {'docId': 1023584516,
   'friendlyName': 'LDFNP4159_csv',
   'postDatetime': '2024-07-29T14:24:41.000',
   '_links': {'endpoint': {'href': 'https://api.ercot.com/api/public-reports/archive/np4-159-cd?download=1023584516'}}},
  {'docId': 1019143955,
   'friendlyName': 'LDFNP4159_csv',
   'postDatetime': '2024-07-12T14:07:07.000',
   '_links': {'endpoint': {'href': 'https://api.ercot.com/api/public-reports/archive/np4-159-cd?download=1019143955'}}},
  {'docId': 1008228679,
"""


DAM_Shadow_Prices = 'NP4-191-CD'
LDF = 'NP4-159-CD'
DAM_HOURLY_LMPS = 'NP4-183-CD'

INDEX_URL = 'https://api.ercot.com/api/public-reports/archive'

LDF_url = f"{INDEX_URL}/{LDF}"
DAM_shadow_url = f"{INDEX_URL}/{DAM_Shadow_Prices}"
DAM_HOURLY_LMPS_url = f"{INDEX_URL}/{DAM_HOURLY_LMPS}"

authorization = 'eyJhbGciOiJSUzI1NiIsImtpZCI6Ilg1ZVhrNHh5b2pORnVtMWtsMll0djhkbE5QNC1jNTdkTzZRR1RWQndhTmsiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3MjY4MDI0MTksIm5iZiI6MTcyNjc5ODgxOSwidmVyIjoiMS4wIiwiaXNzIjoiaHR0cHM6Ly9lcmNvdGIyYy5iMmNsb2dpbi5jb20vNmRmMTdhZmEtMWIzNi00OTlhLTgzZjctNTY3NzlhZDBiOWE2L3YyLjAvIiwic3ViIjoiMzk4NGMyNjktYTc4Mi00NTJlLWJhNGEtYmU3Zjc4NTZhMDBmIiwiYXVkIjoiZmVjMjUzZWEtMGQwNi00MjcyLWE1ZTYtYjQ3OGJhZWVjZDcwIiwibm9uY2UiOiIwMTkyMGQzOS01NmU2LTc5ZDgtYTg1My04NjAyMzQzZjUyNmQiLCJpYXQiOjE3MjY3OTg4MTksImF1dGhfdGltZSI6MTcyNjc5ODc2NSwib2lkIjoiMzk4NGMyNjktYTc4Mi00NTJlLWJhNGEtYmU3Zjc4NTZhMDBmIiwiZW1haWxzIjpbInN2azIxMjBAY29sdW1iaWEuZWR1Il0sImdpdmVuX25hbWUiOiJTYW0iLCJmYW1pbHlfbmFtZSI6IktvcnRjaG1hciIsInRmcCI6IkIyQ18xX1BVQkFQSS1GTE9XIn0.ZAFwNQxnQ4kcpzifksrFeBgyoBnOLIKhb11qrN8Rjx_HtQCUrS-lQibGkyIhs5_gd_UvRGCsLxqLfeDSndRH--ryW8Gok1pd8LQ0ETNMwrR-FUo-sBIg0-OOKdAaPlijq7hkxp3FstrkkZA2HLzwSvh9c9dDKgQ3cXqaspQfsq92hjL8ztL377hd5QphGrZMvMUD3XLCrsNEYqG1SrH6aAe3hEbJTUG8b3DV4g0tJRcyjgZgDvaz61lYuynTpP_8O0xmYxhDYypG9eJds_TwOjQRJtwo_8dkOldPJCJSLiZl-ABPkfUfqg_CGmpgMbZuxQxuBm8t8hlKNefDp4vFMg'
subscription_key = '2d2148b6788c4630a0979d2cbeb6e149'

# Limit time between 2019
request_params = {
    'postDatetimeFrom': '2019-01-01T00:00',
    'postDatetimeTo': '2024-01-01T00:00',
    'size': 1000, # Get all links in one page
}

def refresh_access_token(refresh_token):
    # Authorization URL for signing into ERCOT Public API account
    AUTH_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"

    data = {    
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'response_type': 'id_token',
        'scope': 'openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access',
        'client_id': 'fec253ea-0d06-4272-a5e6-b478baeecd70',
    }

    # Sign In/Authenticate
    auth_response = requests.post(AUTH_URL, data=data)

    # Retrieve access token
    token_response = auth_response.json()  # access_token, refresh_token, id_token
    return token_response    

def get_access_token():
    USERNAME = "svk2120@columbia.edu"
    PASSWORD = "FHpedYrxiSAB2n6"

    # Authorization URL for signing into ERCOT Public API account
    AUTH_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"

    data = {
        'username': USERNAME,
        'password': PASSWORD,
        'grant_type': 'password',
        'response_type': 'id_token',
        'scope': 'openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access',
        'client_id': 'fec253ea-0d06-4272-a5e6-b478baeecd70',
    }

    # Sign In/Authenticate
    auth_response = requests.post(AUTH_URL, data=data)

    # Retrieve access token
    token_response = auth_response.json()  # access_token, refresh_token, id_token
    return token_response

def index_request(url, access_token):
    auth_headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'authorization': 'Bearer ' + access_token['access_token']
    }

    all_matches = []
    for i in [1, 2]: # Only two pages of requests
        request = requests.get(url, headers=auth_headers, params=dict(**request_params, page=i))
        request.raise_for_status()
        data = request.json()
        archives = data.get('archives', [])
        all_matches += [{
            'name': a['friendlyName'],
            'link': a['_links']['endpoint']['href'],
            'id': a['docId'],
        } for a in archives]

    return all_matches


def download_dam_index_batch(access_token):
    all_files = index_request(DAM_shadow_url, access_token)
    auth_headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'authorization': 'Bearer ' + access_token['access_token'],
        # 'content-type': 'application/json; charset=UTF-8',
    }    

    docIds = [str(doc['id']) for doc in all_files[0:3]]
    docIds = ["1037369000", "1037095438"]
    data = {
        'docIds': docIds,
    }
    url = DAM_shadow_url + '/download'

    req = requests.post(url, headers=auth_headers, json=data)
    req.raise_for_status()
    with open('thing.zip', "wb") as f:
        f.write(req.content)


def download_dam_index(access_token):
    all_files = index_request(DAM_shadow_url, access_token)
    auth_headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'authorization': 'Bearer ' + access_token['access_token']
    }

    print(f'Found {len(all_files)}')
    for i, archive in enumerate(all_files):
        name = archive['link'].split('?')[1].split('=')[1]
        print(f'downloading {i}/{len(all_files)}')
        if i < 655:
            continue

        time.sleep(2)
        csv = requests.get(archive['link'], headers=auth_headers)
        csv.raise_for_status()
        with open('dam_shadow/' + name + '.csv', "w") as f:
            f.write(csv.text)

def download_dam_lmps_index(access_token):
    all_files = index_request(DAM_HOURLY_LMPS_url, access_token)
    auth_headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'authorization': 'Bearer ' + access_token['access_token']
    }

    print(f'Found {len(all_files)}')
    for i, archive in enumerate(all_files):
        name = archive['link'].split('?')[1].split('=')[1]
        print(f'downloading {i}/{len(all_files)}')
        time.sleep(2)
        csv = requests.get(archive['link'], headers=auth_headers)
        csv.raise_for_status()
        with open('dam_lmps/' + name + '.csv', "w") as f:
            f.write(csv.text)

def download_ldf_index(access_token):
    all_files = index_request(LDF_url, access_token)
    auth_headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'authorization': 'Bearer ' + access_token['access_token']
    }

    print(f'Found {len(all_files)}')
    for i, archive in enumerate(all_files):
        name = archive['link'].split('?')[1].split('=')[1]
        print(f'downloading {i}/{len(all_files)}')
        time.sleep(2)
        csv = requests.get(archive['link'], headers=auth_headers)
        csv.raise_for_status()
        with open('ldf/' + name + '.csv', "w") as f:
            f.write(csv.text)


def do_downloads(access_token):
    download_dam_index(access_token)
    download_dam_lmps_index(access_token)
    download_ldf_index(access_token)


def ldf_download(access_token):
    auth_headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'authorization': 'Bearer ' + access_token['access_token']
    }
    request = requests.get('https://api.ercot.com/api/public-reports/archive/np4-159-cd?download=1023584516', headers=auth_headers)
    return request.text


def time_ldf_download(access_token):
    start = time.time()
    t = ldf_download(access_token)
    end = time.time()
    print('end - start', end - start)
    return t

# LDF Format:
# LdfDate,LdfHour,SubStation,DistributionFactor,LoadID,MVARDistributionFactor,MRIDLoad,DSTFlag
# 08/06/2024,01:00,MIRAGE,0.2,01NPS-XFM-0001,0,6f09b475-c893-40f6-bcc8-3427ca4c9ed4,N
# 08/06/2024,01:00,SNOOK,3.1,050,0.540999948978424,{356A1C81-051D-432E-A1E7-2B63E50840C3},N

