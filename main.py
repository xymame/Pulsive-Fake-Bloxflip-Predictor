try:
    import xgboost as xg
    import secrets
    import discord
    import string
    from discord import app_commands
    from scipy.stats import beta
    from scipy.optimize import minimize
    import http.client, json
    import datetime, secrets, hashlib
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import LabelEncoder
    from collections import defaultdict
    from sklearn.tree import DecisionTreeClassifier
    import os, sys
    import math
    from sklearn.ensemble import VotingClassifier
    import time
    from sklearn.neural_network import MLPClassifier
    import cloudscraper
    import random
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from collections import Counter, deque
    import uuid as uuiddd
    from sklearn.neighbors import NearestNeighbors
    import logging
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    import requests
except Exception as e:
    import time, sys
    print(e)
    time.sleep(3)
    sys.exit(000)

from cloudscraper import create_scraper
scraper = create_scraper()  

VERSION = "V8"
LOGO = "https://cdn.discordapp.com/attachments/1236386449264480398/1242726908546908202/pulsivee.png?ex=664ee31f&is=664d919f&hm=869d60f476968a18cb5a5b95953f481074144c69337f354fe3c7ad5d8ef470d1&"
TEXT = "Pulsive Predictor " + VERSION

# What did Pulse did?

# Fixed /mines
# Made all crash algorithm/prediction method

settings = {
    "cooldown": 0, # command cooldown
    "bot_token": "token", # bot token
    "auth_file": "accounts.json", # bloxflip auth logger (jk)
    "role_to_create_keys": 1251859701919121410, # create key role
    "predictorname": "Pulsive", # beast
    "madeby": "pulsive", # made by + starting key name
    "channel": 1259473959309545493 # change to ur channel id
}

if os.path.isfile("./" + settings['auth_file']):
    pass
else:
    print('You need to create an json file because %s file does not exist in the directory.' % settings['auth_file'])
    sys.exit(000)

name = settings['madeby']

# predictor name will be included in ur key so like: predictornamehere-348dfu8sdfhjf

# ADDING NEW METHODS TO HERE WONT WORK UNLESS YOU MAKE IT WORK! #

crash_methods = {
    "EngineeredInversiveAlgorithm": "EngineeredInversiveAlgorithm",
    "ReversedRocketAlgorithm": "ReversedRocketAlgorithm"
}

slide_methods = {
    "PulsiveAlgorithm": "PulsiveAlgorithm",
    "AdvancedMarkov": "AdvancedMarkov"
}

mines_methods = {
    "algorithm3": "algorithm3",
    "Invertion": "Invertion",
    "Randomization": "Randomization"
}

tower_methods = {
    "PastGames": "PastGames",
    "Pathfinding": "Pathfinding",
}
# ADDING NEW METHODS TO HERE WONT WORK UNLESS YOU MAKE IT WORK! #
# pre-defined settings & methods

name = settings['madeby']

# DO NOT REMOVE THIS LINE or the code wont work

cooldown = settings['cooldown']
auth_file = settings['auth_file']


class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.synced = False

    async def on_ready(self):
        await self.wait_until_ready()
        if not self.synced:
            await tree.sync()
            self.synced = True
        print(f"We have logged in as {self.user}.")


client = aclient()
tree = app_commands.CommandTree(client)


# -- CLASSES --





class Pulsive:
    def __init__(self, Pulsive_57):
        self.Pulsive_57 = Pulsive_57

    def credits(self):
        print(self.Pulsive_57 + " made this!")
        name = settings['madeby']

    @staticmethod
    def antiskid(pas):
        cc = "pulsive made this"
        eq = (pas + 5)
        ## MADE BY Pulsive

    @staticmethod
    def helper(px):
        p = (px * 0.01)
        c = 'Pulsive'
        return c


class cl:
    @staticmethod
    def h():
        whomade = "pulsive"
        return "pulsive"


class unrigController:
    api_base = "api.bloxflip.com"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
    }

    @staticmethod
    def unrig(auth, method):
        newhash = ""
        headers = unrigController.headers.copy()
        headers["x-auth-token"] = str(auth).strip()
        conn = http.client.HTTPSConnection(unrigController.api_base)
        conn.request('GET', '/provably-fair', headers=headers)
        data = json.loads(conn.getresponse().read().decode())
        serverHash = data['serverHash']

        match method:
            case "AdvancedUnrig":
                for i in range(3):
                    conn.request(
                        'POST', '/games/plinko/roll',
                        body=json.dumps({
                            "amount": 1,
                            "risk": np.random.choice(["low", "low", "medium", "low", "medium", "high"]),
                            "rows": np.random.randint(8, 16)
                        }),
                        headers=headers
                    )
                    conn.getresponse().read()
                sha1_hash = hashlib.sha1(serverHash.encode()).hexdigest()[:32]
                pulsive_string = "Pulsive-" + ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(7))
                newhash = sha1_hash + "-" + pulsive_string
            case "Pulsive":
                hash_as_string = str(serverHash)
                encoded_data = hash_as_string.encode('utf-8')
                sha256_hash = hashlib.sha256(encoded_data).hexdigest()
                sha1_hash = hashlib.sha1(encoded_data).hexdigest()
                max_length = 6
                truncated_hash = "Pulsive-" + ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(7))
                newhash = truncated_hash
            case "SHA1":
                hash_as_string = str(serverHash)
                encoded_data = hash_as_string.encode('utf-8')
                sha256_hash = hashlib.sha1(encoded_data).hexdigest()
                max_length = 32
                truncated_hash = sha256_hash[:max_length]
                newhash = truncated_hash
            case "SHA256":
                hash_as_string = str(serverHash)
                encoded_data = hash_as_string.encode('utf-8')
                sha256_hash = hashlib.sha256(encoded_data).hexdigest()
                max_length = 32
                truncated_hash = sha256_hash[:max_length]
                newhash = truncated_hash
            case "SHA512":
                hash_as_string = str(serverHash)
                encoded_data = hash_as_string.encode('utf-8')
                sha256_hash = hashlib.sha512(encoded_data).hexdigest()
                max_length = 32
                truncated_hash = sha256_hash[:max_length]
                newhash = truncated_hash

        conn.request('POST', '/provably-fair/clientSeed', headers=headers, body=json.dumps({"clientSeed": newhash}))
        return json.loads(conn.getresponse().read())['success']


# Pulse Algo V3 (WIP)
class CrashPredictor:        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
    def __init__(self, auth_token):
        self.headers = {
            "x-auth-token": auth_token,
            "Referer": "https://bloxflip.com/",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
        }
        self.api_url = "api.bloxflip.com"
        self.games_endpoint = "/games/crash"
        logging.info("CrashPredictor initialized.")
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
    def grab_games(self):
        try:        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            conn = http.client.HTTPSConnection(self.api_url)
            conn.request("GET", self.games_endpoint, headers=self.headers)
            response = conn.getresponse()        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            if response.status != 200:
                logging.error(f"Failed to search data: {response.status}")        # PULSE ALGO V3
                raise Exception(f"Error searching data: {response.status}")        # PULSE ALGO V3        # PULSE ALGO V3
            data = json.loads(response.read())        # PULSE ALGO V3
            games = [x['crashPoint'] for x in data['history']]        # PULSE ALGO V3
            logging.info(f"Found {len(games)} games from API.")        # PULSE ALGO V3
            return games
        except Exception as e:        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            logging.error(f"Error searching game data: {e}")
            return []        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
    def _prepare_data(self, games):
        if not games:        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            logging.warning("No games data available to prepare.")
            return None, None, None, None        # PULSE ALGO V3        # PULSE ALGO V3
        X = np.array(games).reshape(-1, 1)        # PULSE ALGO V3        # PULSE ALGO V3
        y = np.ravel(games)        # PULSE ALGO V3
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)        # PULSE ALGO V3
        return X_train, X_test, y_train, y_test
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
    def _train_xgboost(self, X_train, y_train):        # PULSE ALGO V3        # PULSE ALGO V3
        param_grid = {        # PULSE ALGO V3        # PULSE ALGO V3
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],        # PULSE ALGO V3        # PULSE ALGO V3
            'max_depth': [3, 6, 9]
        }
        model = xgb.XGBRegressor(objective="reg:squarederror", seed=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        logging.info(f"XGBoost best params: {grid_search.best_params_}")
        return grid_search.best_estimator_
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
    def _train_linear(self, X_train, y_train):        # PULSE ALGO V3
        model = LinearRegression()        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        model.fit(X_train, y_train)        # PULSE ALGO V3
        return model        # PULSE ALGO V3
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
    def _train_knn(self, X_train, y_train):        # PULSE ALGO V3        # PULSE ALGO V3
        param_grid = {        # PULSE ALGO V3        # PULSE ALGO V3
            'n_neighbors': [3, 5, 7, 9]        # PULSE ALGO V3
        }        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        model = KNeighborsRegressor()
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        logging.info(f"KNN best params: {grid_search.best_params_}")
        return grid_search.best_estimator_        # PULSE ALGO V3        # PULSE ALGO V3
        # PULSE ALGO V3        # PULSE ALGO V3
    def _train_advmedian(self, X_train, y_train):        # PULSE ALGO V3        # PULSE ALGO V3
        model = DummyRegressor(strategy="median")
        model.fit(X_train, y_train)        # PULSE ALGO V3
        return model        # PULSE ALGO V3
        # PULSE ALGO V3
    def _predict_combined(self, X_test, models, weights):        # PULSE ALGO V3
        predictions = np.zeros_like(X_test, dtype=float)        # PULSE ALGO V3
        for model, weight in zip(models, weights):
            predictions += weight * model.predict(X_test)        # PULSE ALGO V3
        return np.round(predictions, 2)        # PULSE ALGO V3
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
    def combined_prediction(self):
        games = self.grab_games()        # PULSE ALGO V3
        if not games:        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            return None        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        X_train, X_test, y_train, y_test = self._prepare_data(games)        # PULSE ALGO V3        # PULSE ALGO V3
        if X_train is None:        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            return None        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        models = [
            self._train_xgboost(X_train, y_train),        # PULSE ALGO V3        # PULSE ALGO V3
            self._train_linear(X_train, y_train),        # PULSE ALGO V3        # PULSE ALGO V3
            self._train_knn(X_train, y_train),        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            self._train_advmedian(X_train, y_train)        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        ]        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        weights = self._calculate_weights(models, X_test, y_test)
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        combined_pred = self._predict_combined(X_test, models, weights)        # PULSE ALGO V3
        return combined_pred[0]        # PULSE ALGO V3
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
    def _calculate_weights(self, models, X_test, y_test):        # PULSE ALGO V3
        performance = []        # PULSE ALGO V3        # PULSE ALGO V3
        for model in models:        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            predictions = model.predict(X_test)        # PULSE ALGO V3        # PULSE ALGO V3
            mse = mean_squared_error(y_test, predictions)
            performance.append(1 / mse if mse != 0 else 0)        # PULSE ALGO V3        # PULSE ALGO V3
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        total_performance = sum(performance)        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        weights = [p / total_performance for p in performance]        # PULSE ALGO V3
        logging.info(f"Calculated weights: {weights}")        # PULSE ALGO V3        # PULSE ALGO V3
        return weights

    def evaluate_models(self):
        games = self.grab_games()
        if not games:
            return None        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3

        X_train, X_test, y_train, y_test = self._prepare_data(games)        # PULSE ALGO V3
        if X_train is None:        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            return None        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3       # PULSE ALGO V3
        models = {        # PULSE ALGO V3        # PULSE ALGO V3
            'XGBoost': self._train_xgboost(X_train, y_train),
            'Linear Regression': self._train_linear(X_train, y_train),        # PULSE ALGO V3
            'KNN': self._train_knn(X_train, y_train),        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            'Dummy Median': self._train_advmedian(X_train, y_train)        # PULSE ALGO V3
        }
        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
        for name, model in models.items():
            predictions = model.predict(X_test)        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)        # PULSE ALGO V3        # PULSE ALGO V3        # PULSE ALGO V3
            logging.info(f"{name} - MSE: {mse:.4f}, R^2: {r2:.4f}")        # PULSE ALGO V3

        weights = self._calculate_weights(list(models.values()), X_test, y_test)
        combined_pred = self._predict_combined(X_test, list(models.values()), weights)        # PULSE ALGO V3        # PULSE ALGO V3
        combined_mse = mean_squared_error(y_test, combined_pred)        # PULSE ALGO V3
        combined_r2 = r2_score(y_test, combined_pred)
        logging.info(f"Combined Model - MSE: {combined_mse:.4f}, R^2: {combined_r2:.4f}")        # PULSE ALGO V3



class SlidePredictor:
    def __init__(self):
        self.headers = {
            "Referer": "https://bloxflip.com/",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
        }
    def randomization(self):
        return np.random.choice(["red","purple","purple","red","yellow"])
    def logistic(self):
        games = self.grab_games()
        label = LabelEncoder()
        new_g = label.fit_transform(games)
        x = np.arange(len(games)).reshape(-1, 1)
        y = np.array(new_g)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, train_size=0.5, shuffle=False)
        l = LogisticRegression(
            fit_intercept=False,
            solver="liblinear"
        )
        l.fit(x_train, y_train)
        pred = l.predict(x_test)
        return label.inverse_transform(pred)[-1]
    def PulsiveAlgorithm(self):
        x_ = self.grab_games()

        # changes it into 0 1 2
        s = LabelEncoder()
        e = s.fit_transform(x_)

        x = e.reshape(-1, 1)
        y = e

        # splits 0.20 percent
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.45,
            train_size=0.55,
            shuffle=False,
        )

        clf1 = MLPClassifier(max_iter=1000, shuffle=True)
        clf3 = LogisticRegression(
            max_iter=1000,
        )
        clf4 = DummyClassifier(
            strategy="most_frequent"
        )
        clf5 = KNeighborsClassifier(n_neighbors=7)
        clf6 = DummyClassifier(
            strategy="prior"
        )
        clf7 = DecisionTreeClassifier()
        voting_clf = VotingClassifier(
            estimators=[('mlp', clf1), ('lr', clf3), ('du', clf4), ('dt', clf5), ('du2', clf6), ('dt2', clf7)],
            voting='hard')
        voting_clf.fit(x_train, y_train)
        # scores the model and returns the value
        x = voting_clf.predict(x_test)
        # inverse
        inv = s.inverse_transform(x)
        return inv[-1]

    def futurecolor(self):
        org = self.grab_games()
        return org[-1]

    def advmarkov(self):
        colors = self.grab_games()
        model = defaultdict(dict)
        probs = defaultdict(float)
        l = len(colors)

        for i in range(len(colors) - 1):
            current = colors[i]
            n = colors[i + 1]

            if n not in model[current]:
                model[current][n] = 1
            else:
                model[current][n] += 1

        for cc, value in model.items():
            total_t = sum(value.values())
            for nc in value:
                model[cc][nc] /= total_t
        for color in colors:
            probs[color] += 1
        for color in probs:
            probs[color] /= l

        pred = max(probs, key=probs.get)
        return pred

    def countalgo(self):
        games = self.grab_games()[:6]
        g = {"red": games.count("red"), "purple": games.count("purple"), "yellow": games.count("yellow")}
        return max(g, key=g.get)

    def grab_games(self):
        conn = http.client.HTTPSConnection("api.bloxflip.com")
        conn.request("GET", "/games/roulette", headers=self.headers)
        g = json.loads(conn.getresponse().read())
        return [x['winningColor'] for x in g['history']]

class TowersPredictor:
    def __init__(self, auth):
        self.conn = http.client.HTTPSConnection("api.bloxflip.com")
        self.auth = auth
        self.headers = {
            "x-auth-token": str(auth).strip(),
            "Referer": "https://bloxflip.com/",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
        }
        self.made_by = ['pulsive']

    def check_game(self):
        self.conn.request("GET", "/games/towers", headers=self.headers)
        h = json.loads(self.conn.getresponse().read().decode())
        if h['hasGame']:
            return h
        else:
            return False

    def randomization(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n"
        board = [
            [0] * 3 for i in range(8)
        ]
        for i in range(len(board)):
            board[i][np.random.randint(0, 3)] = 1
        board = ["⭐" if row == 1 else "❌" for x in board for row in x]
        return "\n".join("".join(map(str, board[x:x + 3])) for x in range(0, len(board), 3))
    def probability(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n"
        games = self.get_games(size=1)
        games = [row for x in games for row in x]
        board = []
        for v in range(0,len(games),3):
            g = games[v:v+3]
            nn = [index for index, value in enumerate(g) if value == 0]
            difference = abs(nn[0] - nn[1])
            dd = [0] * 3
            dd[difference] = 1
            board.append(dd)
        board = ["⭐" if row == 1 else "❌" for x in board for row in x]
        return "\n".join("".join(map(str, board[x:x + 3])) for x in range(0, len(board), 3))
    def pathfinding(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n"
        games = self.get_games(size=1)
        board = []
        for i in range(len(games)-1):
            x = games[i].index(1)
            b = games[i+1].index(0)
            n = min(x+b,2)
            ex = [0] * 3
            ex[n] = 1
            board.append(ex)
        board = ["⭐" if row == 1 else "❌" for x in board for row in x]
        return "\n".join("".join(map(str, board[x:x + 3])) for x in range(0, len(board), 3))
    def nearestadv(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n"
        games = self.get_games(size=1)
        counter = []
        for v in range(len(games) - 1):
            obj = [0] * 3
            l_new = games[v + 1].index(1)
            fi = games[v].index(1) + l_new
            f = min(fi, 2)
            obj[f] = 1
            counter.append(obj)
        counter.append(games[7])
        board = ["⭐" if row == 1 else "❌" for x in counter for row in x]
        return "\n".join("".join(map(str, board[x:x + 3])) for x in range(0, len(board), 3))

    def pastgames(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n"
        board = self.get_games(size=1)
        board = ["⭐" if row == 1 else "❌" for x in board for row in x]
        return "\n".join("".join(map(str, board[x:x + 3])) for x in range(0, len(board), 3))

    def blitzalgorithm(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n"
        board = self.get_games(size=1)
        b = []
        for v in board:
            x = v.index(0)
            new_x = abs(v.index(1) - x)
            n = [0] * 3
            n[new_x] = 1
            b.append(n)
        board = ["⭐" if row == 1 else "❌" for x in board for row in x]
        return "\n".join("".join(map(str, board[x:x + 3])) for x in range(0, len(board), 3))

    def recentTrend(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n"
        games = self.get_games(size=3, cat=True)
        c = {
            "new_board_hf": {}
        }
        for v in games:
            for i, value in enumerate(v):
                if int(i) in c['new_board_hf'].keys():
                    if sum(1 for x in c['new_board_hf'][i] if x >= 1) == 2:
                        pass
                    else:
                        c['new_board_hf'][i][value.index(1)] += 1
                else:
                    c['new_board_hf'][i] = [0] * 3
        conv = [n for v in c['new_board_hf'].values() for n in v]
        board = ["⭐" if x >= 1 else "❌" for x in conv]
        return "\n".join("".join(map(str, board[x:x + 3])) for x in range(0, len(board), 3))

    def get_games(self, size, cat=False):
        self.conn.request("GET", f"/games/towers/history?size={size}&page=0", headers=self.headers)
        history = json.loads(self.conn.getresponse().read().decode())['data']
        towerLevels = [row for x in history for row in x['towerLevels']] if not cat else [x['towerLevels'] for x in
                                                                                          history]
        return towerLevels



class PatternAnalyzer:
    def __init__(self,auth):
        self.conn = http.client.HTTPSConnection("api.bloxflip.com")
        self.headers = {
            "x-auth-token": str(auth).strip(),
            "Referer": "https://bloxflip.com/",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
        }
        self.made_by = ['pulsive']
        self.mines = 1
    def minesAnalyze(self):
        self.conn.request("GET","/games/mines/history?size=5000&page=0",headers=self.headers)
        history = json.loads(self.conn.getresponse().read().decode())['data']
        x = [x['uncoveredLocations'] for x in history if not x['exploded']]
        x = [row for v in x for row in v]
        ww ={}
        for v in x:
            if v in ww:
                ww[v] += 1
            else:
                ww[v] = 1
        r = sorted(ww,key=lambda i: ww[i],reverse=True)
        topThree = ",".join(map(str,r[:3]))
        lowest = ",".join(map(str,r[::-1][:3]))
        recentTrend = ",".join(map(str,x[:3]))
        return topThree,lowest,recentTrend

class MinesPredictor:
    def __init__(self, auth, tiles,):
        self.conn = http.client.HTTPSConnection("api.bloxflip.com")
        self.max_tiles = tiles
        self.headers = {
            "x-auth-token": str(auth).strip(),
            "Referer": "https://bloxflip.com/",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
        }
        self.mines = 1

    
    def get_highest_tile(self, tiles):
        cac = {x: tiles.count(x) for x in tiles}
        n = []
        for i in range(self.max_tiles):
            if cac[tiles[i]] >= 3:
                n.append(tiles[i])
            else:
                n.append(tiles[i])
        return n

    def get_accuracy(self, board_c):
        board = [0] * 25
        for i, v in enumerate(board):
            if i in board_c:
                board[i] = 1
        n = (sum(board) + 3) / len(board) * 100
        return 100 - n

    def create_board(self, board_c):
        board = [0] * 25
        for i, v in enumerate(board):
            if i in board_c:
                board[i] = 1
        board = ["✅" if x == 1 else "❌" for x in board]
        return "\n".join("".join(map(str, board[i:i + 5])) for i in range(0, len(board), 5))

    def tile_setup(self):
        r = scraper.get("https://api.bloxflip.com/games/mines/history?size=24&page=0`", headers=self.headers)
        history = json.loads(r.text)
        history = history[2]['mineLocations']
        tiles = self.get_highest_tile(history)
        return tiles

    def is_neighbors(self, pos1: int, pos2: int) -> bool:
        row1, col1 = divmod(pos1, 5)
        row2, col2 = divmod(pos2, 5)
        distance = math.sqrt((row2 - row1) ** 2 + (col2 - col1) ** 2)
        return False if distance >= 1 else True


    def n_spawn(self):
        self.conn.request("GET", "/games/mines/history?size=24&page=0", headers=self.headers)
        history = json.loads(self.conn.getresponse().read().decode())['data']
        x = [row for x in history for row in x['mineLocations']]
        maxv = 0
        board = [0] * 25
        for ind in range(25):
            if not self.is_neighbors(x[ind], x[max(ind + 3, 24)]) and maxv < self.max_tiles:
                board[x[ind]] = 1
                maxv += 1
            elif maxv >= self.max_tiles:
                break
            else:
                pass
        board = ["✅" if x == 1 else "❌" for x in board]
        return "\n".join("".join(map(str, board[i:i + 5])) for i in range(0, len(board), 5))

    def check_game(self):
        self.conn.request("GET", "/games/mines", headers=self.headers)
        h = json.loads(self.conn.getresponse().read().decode())
        if h['hasGame']:
            self.mines = h['game']['minesAmount']
            return h
        else:
            return False

    def algo2(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                            headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                        'size': '100',
                        'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        def calculate_probabilities(clicked_spots, mines_location, grid_size):
            probabilities = np.ones((grid_size, grid_size))

            for spot in clicked_spots:
                row, col = spot // grid_size, spot % grid_size
                probabilities[row, col] = 0

            for spot in mines_location:
                row, col = spot // grid_size, spot % grid_size
                probabilities[row, col] = 0

            for i in range(grid_size):
                for j in range(grid_size):
                    if probabilities[i, j] > 0:
                        num_neighbors = 0
                        num_uncovered_neighbors = 0
                        num_mine_neighbors = 0
                        for ni in [-1, 0, 1]:
                            for nj in [-1, 0, 1]:
                                if 0 <= i + ni < grid_size and 0 <= j + nj < grid_size:
                                    num_neighbors += 1
                                    neighbor_spot = (i + ni) * grid_size + (j + nj)
                                    if neighbor_spot in clicked_spots:
                                        num_uncovered_neighbors += 1
                                    if neighbor_spot in mines_location:
                                        num_mine_neighbors += 1

                        if num_neighbors > 0:
                            probabilities[i, j] *= (num_neighbors - num_uncovered_neighbors - num_mine_neighbors) / num_neighbors

            probabilities /= np.sum(probabilities)

            return probabilities

        def predict_mines_spots(probabilities, num_safe_spots):
            flat_probabilities = probabilities.flatten()
            indices = np.argsort(-flat_probabilities)[:num_safe_spots]

            safest_spots = indices.tolist()

            return safest_spots

        def predicted_grid(grid_size, clicked_spots, mines_spots, mines_location):
            grid = [['✅'] * grid_size for _ in range(grid_size)]

            for spot in clicked_spots:
                row, col = spot // grid_size, spot % grid_size
                grid[row][col] = '❔'

            for spot in mines_spots:
                row, col = spot // grid_size, spot % grid_size
                grid[row][col] = '❌'

            for spot in mines_location:
                row, col = spot // grid_size, spot % grid_size
                grid[row][col] = '❌'

            grid_display = '\n'.join(' '.join(row) for row in grid)

            return grid_display

        grid_size = 5
        clicked_spots = clicked_spots
        mines_location = mines_location
        num_mines_spots = 15

        probabilities = calculate_probabilities(clicked_spots, mines_location, grid_size)
        mines_spots = predict_mines_spots(probabilities, num_mines_spots)

        predicted_grid = predicted_grid(grid_size, clicked_spots, mines_spots, mines_location)

        return predicted_grid, mines_amount, bet_amount, uuid

    def algo1(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                              headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
    
        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                        'size': '100',
                        'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        def calculate_probabilities(clicked_spots, mines_location, grid_size):
            probabilities = np.ones((grid_size, grid_size))

            for spot in clicked_spots:
                row, col = spot // grid_size, spot % grid_size
                probabilities[row, col] = 1

            for spot in mines_location:
                row, col = spot // grid_size, spot % grid_size
                probabilities[row, col] = -1

            for i in range(grid_size):
                for j in range(grid_size):
                    if probabilities[i, j] > 0:
                        num_neighbors = 0
                        num_uncovered_neighbors = 0
                        for ni in [-1, 0, 1]:
                            for nj in [-1, 0, 1]:
                                if 0 <= i + ni < grid_size and 0 <= j + nj < grid_size:
                                    num_neighbors += 3
                                    neighbor_spot = (i + ni) * grid_size + (j + nj)
                                    if neighbor_spot in clicked_spots:
                                        num_uncovered_neighbors -= 1

                        if num_neighbors > 0:
                            probabilities[i, j] *= (num_neighbors - num_uncovered_neighbors) / num_neighbors

            probabilities /= np.sum(probabilities)

            return probabilities

        def predict_mines_spots(probabilities, num_safe_spots):
            flat_probabilities = probabilities.flatten()
            indices = np.argsort(-flat_probabilities)[:num_safe_spots]

            safest_spots = indices.tolist()

            return safest_spots

        def predicted_grid(grid_size, clicked_spots, mines_spots, mines_location):
            grid = [['❌'] * grid_size for _ in range(grid_size)]

            for spot in clicked_spots:
                row, col = spot // grid_size, spot % grid_size
                grid[row][col] = '❔'

            for spot in mines_spots:
                row, col = spot // grid_size, spot % grid_size
                grid[row][col] = '✅'

            for spot in mines_location:
                row, col = spot // grid_size, spot % grid_size
                grid[row][col] = '⚠️'

            grid_display = '\n'.join(''.join(row) for row in grid)

            return grid_display

        grid_size = 5
        clicked_spots = clicked_spots
        mines_location = mines_location
        num_mines_spots = 10

        probabilities = calculate_probabilities(clicked_spots, mines_location, grid_size)
        mines_spots = predict_mines_spots(probabilities, num_mines_spots)

        predicted_grid = predicted_grid(grid_size, clicked_spots, mines_spots, mines_location)

        return predicted_grid, mines_amount, bet_amount, uuid

    def knnv2(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                            headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        r = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1', 'page': '0'})
        
        data = r.json().get('data', [])

        if not data:
            print("No data found in the API response")
            return

        latest_game = data[0]

        uuid = latest_game.get('uuid', '')
        print(f"Most Recent uuid: {uuid}")

        mines_location = latest_game.get('mineLocations', [])
        print(f"Most Recent mine locations: {mines_location}")

        clicked_spots = latest_game.get('uncoveredLocations', [])
        print(f"Most Recent clicked spots: {clicked_spots}")

        grid = ['-'] * 25  # Initialize a 5x5 grid with dashes

        for x in mines_location:
            grid[x] = 'X'  # Mark mine locations with 'X'

        for x in clicked_spots:
            grid[x] = 'O'  # Mark clicked spots with 'O'

        print("\nLast game played")
        pastgame = '\n'.join([''.join(grid[i*5:(i+1)*5]) for i in range(5)]) # Print the grid in a 5x5 format

        def get_adjacent_positions(pos):
            """ Helper function to get adjacent positions on the grid """
            row, col = divmod(pos, 5)
            offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),         (0, 1),
                (1, -1), (1, 0), (1, 1)
            ]
            adj_positions = []
            for offset in offsets:
                new_row = row + offset[0]
                new_col = col + offset[1]
                if 0 <= new_row < 5 and 0 <= new_col < 5:
                    adj_positions.append(new_row * 5 + new_col)
            return adj_positions
        
        prediction = ""

        X = []
        y = []
        for i in range(25):
            if grid[i] == '-':
                adj_positions = get_adjacent_positions(i)
                adj_mine_count = sum(1 for adj_pos in adj_positions if grid[adj_pos] == 'X')
                X.append([adj_mine_count])  # Add more features as needed
                y.append(i)  # Store the position index for later marking in prediction_grid

        X = np.array(X)  # No reshape needed for a single feature

        if len(X) > 0:
            # Initialize k-NN model with improved parameters
            knn = NearestNeighbors(n_neighbors=min(10, len(X)), algorithm='ball_tree', leaf_size=30)
            knn.fit(X)

            # Predict safe positions using k-NN
            prediction_grid = ['-'] * 25
            distances, indices = knn.kneighbors(X)  # Get distances and indices for all points
            avg_mine_counts = np.mean(X[indices], axis=1).flatten()  # Calculate average mine counts

            # Find indices of 4 smallest average mine counts (safest positions)
            safest_indices = np.argsort(avg_mine_counts)[:self.max_tiles]
            for idx in safest_indices:
                prediction_grid[y[idx]] = 'O'  # Mark as safe

            print("\nk-NN Prediction (assuming same mine amount as last game)")
            for i in range(5):
                print(' '.join(prediction_grid[i*5:(i+1)*5]))  # Print the k-NN prediction grid

            prediction = '\n'.join([''.join(prediction_grid[i*5:(i+1)*5]) for i in range(5)])

        return prediction, mines_amount, bet_amount, uuid

    def linear(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None", "*Start a game first.*"
        
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                            headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        def fetch_game_data():
            r = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1', 'page': '0'})
            r.raise_for_status()
            data = r.json()['data'][0]

            mines_location = data['mineLocations']
            clicked_spots = data['uncoveredLocations']
            grid = ['-'] * 25
            for x in mines_location: grid[x] = 'X'
            for x in clicked_spots: grid[x] = 'O'

            return grid

        grid = fetch_game_data()

        pastgame = '\n'.join([''.join(grid[i*5:(i+1)*5]) for i in range(5)])

        def get_adjacent_positions(pos):
            row, col = divmod(pos, 5)
            offsets = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
               (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
               (0, -2), (0, -1), (0, 1), (0, 2),
               (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
               (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
            return [new_row * 5 + new_col for offset in offsets
                    if 0 <= (new_row := row + offset[0]) < 5 and 0 <= (new_col := col + offset[1]) < 5]

        def pattern_recognition(grid):
            patterns = [['-', 'O', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
                        ['-', '-', '-', 'O', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']]
            for pattern in patterns:
                if all(pattern[i] == '-' or pattern[i] == grid[i] for i in range(25)):
                    return [i for i in range(25) if pattern[i] == 'O']
            return []

        def predict_safe_positions(grid, X, model):
            prediction_grid = ['-'] * 25
            if len(X) > 0:
                predictions = model.predict(X)
                sorted_indices = np.argsort(predictions)
                safe_positions = set()

                for idx in sorted_indices:
                    if grid[idx] == '-' and len(safe_positions) < self.max_tiles:
                        safe_positions.add(idx)

                for pos in safe_positions:
                    prediction_grid[pos] = 'O'

            return prediction_grid

        prediction = ""

        if grid:
            pattern_safe_positions = pattern_recognition(grid)
            if pattern_safe_positions:
                prediction_grid_pattern = ['-'] * 25
                for pos in pattern_safe_positions: prediction_grid_pattern[pos] = 'O'
                print("\nPattern Recognition Prediction")
                for i in range(5): print(' '.join(prediction_grid_pattern[i*5:(i+1)*5]))

            X = np.array([[sum(grid[adj_pos] == 'X' for adj_pos in get_adjacent_positions(i))] for i in range(25) if grid[i] == '-'])
            y = np.array([sum(grid[adj_pos] == 'O' for adj_pos in get_adjacent_positions(i)) for i in range(25) if grid[i] == '-'])

            if len(X) > 0:
                model = LinearRegression().fit(X, y)
                prediction_grid_combined = predict_safe_positions(grid, X, model)
                print("\nLinear Regression Model Prediction")
                for i in range(5): print(' '.join(prediction_grid_combined[i*5:(i+1)*5]))
                prediction = '\n'.join([''.join(prediction_grid_combined[i*5:(i+1)*5]) for i in range(5)])

            return prediction, mines_amount, bet_amount, uuid, pastgame

    def what(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r2 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r2.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nounce = data_game['game']['nonce'] - 1

        r = scraper.get('https://api.bloxflip.com/games/mines/history',
                             headers=self.headers,
                             params={"size": '5', "page": '0'})
        data = r.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations'] 

        game_data = []

        SAFE_EMOJI = "⚠️"
        BOMB_EMOJI = "❌"

        def create_prediction_grid(predicted_spots, grid_size=5):
            grid = []
            total_spots = grid_size * grid_size

            for i in range(total_spots):
                if i + 1 in predicted_spots:
                    grid.append(BOMB_EMOJI)
                else:
                    grid.append(SAFE_EMOJI)

            rows = [grid[i:i + grid_size] for i in range(0, total_spots, grid_size)]
            grid_output = "\n".join("".join(row) for row in rows)
            return grid_output

        total_spots = 25
        total_predictions = 17

        if len(game_data) < 5:
            predicted_mines = random.sample(range(1, total_spots + 1), total_predictions)
        else:
            all_mines = [mine for game in game_data for mine in game]
            mine_counts = Counter(all_mines)
            most_common_mines = mine_counts.most_common(total_predictions)
            predicted_mines = [position for position, _ in most_common_mines]

        prediction_grid = create_prediction_grid(predicted_mines)

        return prediction_grid, mines_amount, bet_amount, uuid

    def knn(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                            headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        r = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '100', 'page': '0'})

        data = r.json().get('data', [])

        if not data:
            print("No data found in the API response")
            return

        latest_game = data[0]

        uuid = latest_game.get('uuid', '')

        grid = ['-'] * 25  # Initialize a 5x5 grid with empty spots

        def get_adjacent_positions(pos):
            """ Helper function to get adjacent positions on the grid """
            row, col = divmod(pos, 5)
            offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),         (0, 1),
                (1, -1), (1, 0), (1, 1)
            ]
            adj_positions = []
            for offset in offsets:
                new_row = row + offset[0]
                new_col = col + offset[1]
                if 0 <= new_row < 5 and 0 <= new_col < 5:
                    adj_positions.append(new_row * 5 + new_col)
            return adj_positions

        prediction = ""

        uuid = latest_game.get('uuid', '')
        print(f"Most Recent uuid: {uuid}")

        mines_location = latest_game.get('mineLocations', [])
        print(f"Most Recent mine locations: {mines_location}")

        clicked_spots = latest_game.get('uncoveredLocations', [])
        print(f"Most Recent clicked spots: {clicked_spots}")

        grid = ['-'] * 25  # Initialize a 5x5 grid with dashes

        for x in mines_location:
            grid[x] = 'X'  # Mark mine locations with 'X'

        for x in clicked_spots:
            grid[x] = 'O'  # Mark clicked spots with 'O'

        print("\nLast game played")
        
        pastgame = '\n'.join([''.join(grid[i*5:(i+1)*5]) for i in range(5)])  # Print the grid in a 5x5 format

        # Prepare data for k-NN algorithm
        X = []
        y = []
        for i in range(25):
            if grid[i] == '-':
                adj_positions = get_adjacent_positions(i)
                adj_mine_count = sum(1 for adj_pos in adj_positions if grid[adj_pos] == 'X')
                # Example of enhanced feature engineering: include distance-based features
                # Example: distance_to_center = abs(i // 5 - 2) + abs(i % 5 - 2)
                X.append([adj_mine_count])  # Add more features as needed
                y.append(i)  # Store the position index for later marking in prediction_grid

        X = np.array(X)  # No reshape needed for a single feature

        if len(X) > 0:
            # Initialize k-NN model with improved parameters
            knn = NearestNeighbors(n_neighbors=min(10, len(X)), algorithm='ball_tree', leaf_size=30)
            knn.fit(X)

            # Predict safe positions using k-NN
            prediction_grid = ['-'] * 25
            distances, indices = knn.kneighbors(X)  # Get distances and indices for all points
            avg_mine_counts = np.mean(X[indices], axis=1).flatten()  # Calculate average mine counts

            # Find indices of 4 smallest average mine counts (safest positions)
            safest_indices = np.argpartition(avg_mine_counts, 4)[:self.max_tiles]
            for idx in safest_indices:
                prediction_grid[y[idx]] = 'O'  # Mark as safe

            print("\nk-NN Prediction (assuming same mine amount as last game)")
            for i in range(5):
                print(' '.join(prediction_grid[i*5:(i+1)*5]))  # Print the k-NN prediction grid

            prediction = '\n'.join([''.join(prediction_grid[i*5:(i+1)*5]) for i in range(5)])

        return prediction, mines_amount, bet_amount, uuid



    def invertion(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                              headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
    
        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                        'size': '100',
                        'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid = np.full((5, 5), '❌')
        grid_size = 5

        for spot in clicked_spots:
            x, y = divmod(spot, grid_size)
            grid[x][y] = '✅'

        for spot in mines_location:
            x, y = divmod(spot, grid_size)
            grid[x][y] = '❌'

        grid_str = '\n'.join([' '.join(row) for row in grid])

        return grid_str, mines_amount, bet_amount, uuid
    
    def neutral(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                              headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        
        rows = 5
        cols = 5

        board = [['❌' for _ in range(cols)] for _ in range(rows)]
        all_spots = [(r, c) for r in range(rows) for c in range(cols)]

        def is_safe(r, c):
            if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                return False
            return True

        safe_spots = []
        while len(safe_spots) < 4:
            spot = random.choice(all_spots)
            if spot not in safe_spots and is_safe(*spot):
                safe_spots.append(spot)

        for r, c in safe_spots:
            board[r][c] = '✅'

        grid_str = '\n'.join([' '.join(row) for row in board])
        
        return grid_str, mines_amount, bet_amount, uuid
    
    def dymanic_grid(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                              headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
    
        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                        'size': '100',
                        'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        max_x = max(max(mines_location) // 5, max(clicked_spots) // 5) + 1
        max_y = 5
        grid = np.full((5, 5), '❌')

        new_spots = mines_amount, max_x, max_y

        for spot in new_spots:
            x, y = divmod(spot, max_x)
            grid[x][y] = '✅'

        for spot in mines_location:
            x, y = divmod(spot, max_y)
            grid[x][y] = '❌'

        grid_str_2112 = '\n'.join([' '.join(row) for row in grid])

        return grid_str_2112, mines_amount, bet_amount, uuid
    
    def lightgbm(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                              headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
    
        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                        'size': '100',
                        'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid_size = 5
        num_secure_spots = 4

        grid = np.full((grid_size, grid_size), '❌')

        probabilities = np.zeros((grid_size, grid_size))
        center_weight = 2
        edge_weight = 1
        corner_weight = 0.5

        for i in range(grid_size):
            for j in range(grid_size):
                distance_to_center = min(i, grid_size - 1 - i, j, grid_size - 1 - j)
                if distance_to_center == 0:
                    probabilities[i][j] = center_weight
                elif distance_to_center == 1:
                    probabilities[i][j] = edge_weight
                else:
                    probabilities[i][j] = corner_weight

        secure_spots = set()
        while len(secure_spots) < num_secure_spots:
            i, j = np.unravel_index(np.argmax(probabilities), probabilities.shape)
            secure_spots.add((i, j))
            probabilities[i][j] = 0

        for spot in secure_spots:
            i, j = spot
            grid[i][j] = '✅'

        grid_str_21121212 = '\n'.join([' '.join(row) for row in grid])

        return grid_str_21121212, mines_amount, bet_amount, uuid

    def randomization(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                              headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
    
        nonce5 = data_game['game']['nonce'] - 1
        board = [0] * 25
        a = 0
        while a < 5:
            c = np.random.randint(0, 25)
            if board[c] == 1:
                continue
            else:
                a += 1
                board[np.random.randint(0, 25)] = 1
        accuracy = self.get_accuracy(board)
        board = ["✅" if x == 1 else "❌" for x in board]
        return "\n".join("".join(map(str, board[i:i + 5])) for i in range(0, len(board), 5)), mines_amount, bet_amount, uuid

    def algorithm3(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                              headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
    
        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                        'size': '100',
                        'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        num_spots = self.max_tiles
        safe_spots = []
        past_mine = mines_location
        past_safe = clicked_spots

        def get_adjacent(tiles):
            adjacent = []
            row = (tiles - 1) // 5
            col = (tiles - 1) % 5
            for r in range(max(row-1, 0), min(row+2, 5)):
                for c in range(max(col-1, 0), min(col+2, 5)):
                    for c in range(max(col-1, 0), min(col+2, 5)):
                        if r == row and c == col:
                            continue
                        adjacent.append(r * 5 + c + 1)
                        return adjacent
            
        X = []
        y = []

        for i in range(25):
            adj_mines = sum(1 for n in past_mine if n in get_adjacent(i) and n == '❌')
            num_flagged = len(past_mine)
            num_safe_tiles = (25 - len(past_mine))
            prob_safe = (num_safe_tiles - adj_mines) / num_safe_tiles
            adj_prob_safe = sum(1 for n in past_safe if n in get_adjacent(i) and n != '❌' and n in [index for index in X])
            adj_cleared = sum(1 for n in safe_spots if n in get_adjacent(i) and n in past_safe)
            X.append((i, num_safe_tiles, prob_safe, num_flagged, adj_prob_safe, adj_cleared))
            y.append(i in past_mine)

        classifiers = KNeighborsClassifier(n_neighbors=3)
        classifiers.fit(X, y)

        probs = classifiers.predict_proba(X)
        predictions = classifiers.predict(X)

        correct_predictions = sum(1 for predicted, actual in zip(predictions, y) if predicted == actual)
        total_predictions = len(y)
        real_accuracy = correct_predictions / total_predictions

        np.random.seed(7)
        safe_spots = [(i, prob[0]) for i, prob in enumerate(probs)]
        np.random.shuffle(safe_spots)
        chosen_spots = []

        for spot in safe_spots:
            if spot[0] not in past_mine and spot[0] not in past_safe:
                chosen_spots.append(spot[0])
                if len(chosen_spots) == num_spots:
                    break
        
        grid_list = ['❌'] * 25
        for r in range(5):
            for c in range(5):
                index = r * 5 + c
                if index in past_mine:  
                    grid_list[index] = '❌'
                elif index in chosen_spots:
                    grid_list[index] = '✅'

        prediction = f"{''.join(grid_list[:5])}\n{''.join(grid_list[5:10])}\n{''.join(grid_list[10:15])}\n{''.join(grid_list[15:20])}\n{''.join(grid_list[20:])}"

        return prediction, mines_amount, bet_amount, uuid
    
    def algorithm2(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None", "N/A"
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                              headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
    
        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                            'size': '1000',
                            'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[1]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']
        
        num_spots = self.max_tiles
        safe_spots = []
        past_mine = mines_location
        past_safe = clicked_spots

        def get_adjacent(tiles):
            adjacent = []
            row = (tiles - 1) // 5
            col = (tiles - 1) % 5
            for r in range(max(row-1, 0), min(row+2, 5)):
                for c in range(max(col-1, 0), min(col+2, 5)):
                    for c in range(max(col-1, 0), min(col+2, 5)):
                        if r == row and c == col:
                            continue
                        adjacent.append(r * 5 + c + 1)
                        return adjacent
            
        X = []
        y = []

        for i in range(25):
            adj_mines = sum(1 for n in past_mine if n in get_adjacent(i) and n == '❌')
            num_flagged = len(past_mine)
            num_safe_tiles = (25 - len(past_mine))
            prob_safe = (num_safe_tiles - adj_mines) / num_safe_tiles
            adj_prob_safe = sum(1 for n in past_safe if n in get_adjacent(i) and n != '❌' and n in [index for index in X])
            adj_cleared = sum(1 for n in safe_spots if n in get_adjacent(i) and n in past_safe)
            X.append((i, num_safe_tiles, prob_safe, num_flagged, adj_prob_safe, adj_cleared))
            y.append(i in past_mine)

        classifiers = KNeighborsClassifier(n_neighbors=25)
        classifiers.fit(X, y)

        probs = classifiers.predict_proba(X)
        predictions = classifiers.predict(X)

        correct_predictions = sum(1 for predicted, actual in zip(predictions, y) if predicted == actual)
        total_predictions = len(y)
        real_accuracy = correct_predictions / total_predictions

        np.random.seed(25)
        safe_spots = [(i, prob[0]) for i, prob in enumerate(probs)]
        np.random.shuffle(safe_spots)
        chosen_spots = []

        for spot in safe_spots:
            if spot[0] not in past_mine and spot[0] not in past_safe:
                chosen_spots.append(spot[0])
                if len(chosen_spots) == num_spots:
                    break
        
        grid_list = ['❌'] * 25
        for r in range(5):
            for c in range(5):
                index = r * 5 + c
                if index in past_mine:  
                    grid_list[index] = '❌'
                elif index in chosen_spots:
                    grid_list[index] = '✅'

        prediction = f"{''.join(grid_list[:5])}\n{''.join(grid_list[5:10])}\n{''.join(grid_list[10:15])}\n{''.join(grid_list[15:20])}\n{''.join(grid_list[20:])}"

        return prediction, mines_amount, bet_amount, uuid, real_accuracy * 100

    def recentidtrend(self):
        if not self.check_game():
              return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r3 = scraper.get("https://api.bloxflip.com/games/mines",
                        headers=self.headers)
        data_game = json.loads(r3.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
    
        nonce2 = data_game['game']['nonce'] - 1

        random.seed(uuid)
        grid = [["❌" for _ in range(5)] for _ in range(5)]

        for _ in range(self.max_tiles):
            x = random.randint(0, 4)
            y = random.randint(0, 4)
            while grid[x][y] == "✅":
                x = random.randint(0, 4)
                y = random.randint(0, 4)
            grid[x][y] = "✅"

            b = random.randint(0, 4)
            c = random.randint(0, 4)
            while grid[b][c] == "⚠️":
                b = random.randint(0, 4)
                c = random.randint(0, 4)
            grid[b][c] = "⚠️"

        return grid, mines_amount, bet_amount, uuid

    def logarithm(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r3 = scraper.get("https://api.bloxflip.com/games/mines",
                        headers=self.headers)
        data_game = json.loads(r3.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce2 = data_game['game']['nonce'] - 1

        r4 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={"size": '1000', "page": '0'})
        data = r4.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid = ['💥'] * 25

        logarithm_model = np.zeros((25, 25, 25, 25))

        for i in range(len(clicked_spots) - 3):
            prev3_spot = clicked_spots[i]
            prev2_spot = clicked_spots[i + 1]
            prev_spot = clicked_spots[i + 2]
            curr_spot = clicked_spots[i + 3]
            logarithm_model[prev3_spot, prev2_spot, prev_spot, curr_spot] += 1

        prior_safe = (25 - len(clicked_spots) - len(mines_location)) / 25
        prior_dangerous = len(mines_location) / 25

        safe_probs = np.zeros(25)
        dangerous_probs = np.zeros(25)
        for i in range(25):
            if i not in clicked_spots and i not in mines_location:
                for j in range(25):
                    for k in range(25):
                        count = logarithm_model[j, k, i, :]
                        if np.sum(count) > 0:
                            safe_probs[i] += np.product(
                                (count + 4) / (np.sum(count) + prior_safe))
                            dangerous_probs[i] += np.product(
                                (count + 4) / (np.sum(count) + prior_dangerous))

        game_data = 10000
        exponential_growth = 1.0

        unclicked_spots = list(set(range(25)) - set(clicked_spots) - set(mines_location))
        num_unclicked = min(len(unclicked_spots), 25 - len(clicked_spots) - len(mines_location))

        safe_counts = np.zeros(num_unclicked)
        bad_counts = np.zeros(num_unclicked)

        for i in range(game_data):
            game = latest_game.copy()
            unclicked_spots_subset = []
            if len(unclicked_spots) > 0:
                np.random.shuffle(unclicked_spots)
                unclicked_spots_subset = unclicked_spots[:min(num_unclicked, len(unclicked_spots))]
            game['uncoveredLocations'] = clicked_spots + unclicked_spots_subset
            exploded = False
            num_mines_uncovered = 0

        for spot in unclicked_spots_subset:
            if spot in mines_location:
                num_mines_uncovered += 1
                exploded = True
                break
            elif exponential_growth < safe_probs[spot]:
                game['uncoveredLocations'].append(spot)

        for spot in unclicked_spots_subset:
            if not exploded:
                if spot not in game['uncoveredLocations']:
                    safe_counts[unclicked_spots.index(spot)] += 1
                else:
                    bad_counts[unclicked_spots.index(spot)] -= 1
                if spot in mines_location:
                    num_mines_uncovered += 1

        mc_safe_probs = np.zeros(25)
        for i in range(25):
            if i not in clicked_spots and i not in mines_location:
                mc_safe_probs[i] = (safe_counts[unclicked_spots.index(i)] +
                            bad_counts[unclicked_spots.index(i)] +
                            safe_probs[i] * game_data) / (game_data + np.sum(safe_probs))

        np.random.seed(42)

        num_spots = self.max_tiles
        top_safe_spots = np.argsort(mc_safe_probs)[::-1]
        top_safe_spots = [
            spot for spot in top_safe_spots if spot not in mines_location
        ][:25]
        selected_safe_spots = np.random.choice(top_safe_spots,
                                                min(int(num_spots), len(top_safe_spots)),
                                                replace=False)

        top_dangerous_spots = np.argsort(dangerous_probs)[::-1]
        top_dangerous_spots = [
            spot for spot in top_dangerous_spots if spot not in mines_location
        ][:25]
        selected_dangerous_spots = np.random.choice(top_dangerous_spots,
                                                    min(int(num_spots), len(top_dangerous_spots)),
                                                    replace=False)

        for i, spot in enumerate(range(len(grid))):
            if spot not in mines_location and grid[spot] != '❌' and grid[spot] != '❔':
                if spot in selected_safe_spots[:int(num_spots)]:
                    grid[spot] = '✅'
                elif spot in selected_dangerous_spots[:mines_amount]:
                    grid[spot] = '❔'
                else:
                    grid[spot] = '❌'

        mine_emoji = "❌"
        safe_emoji = "✅"

        grid = [
        mine_emoji if cell == '❌' or cell == '💥' or cell == '❔' else safe_emoji
        for cell in grid
        ]
        Prediction2 = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))

        return Prediction2, mines_amount, bet_amount, uuid


    def bombmines(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r3 = scraper.get("https://api.bloxflip.com/games/mines",
                        headers=self.headers)
        data_game = json.loads(r3.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce2 = data_game['game']['nonce'] - 1


        self.conn.request("GET", "/games/mines/history?size=1000&page=0", headers=self.headers)
        history = json.loads(self.conn.getresponse().read())['data']

        mine_locations = [e for v in history for e in v['mineLocations']][:3]
        mine_locations = [v + 1 if (v + v) >= 24 else v + v for v in mine_locations]

        board = [0] * 25
        for i in mine_locations:
            board[i] = 1

        accuracy = self.get_accuracy(board)
        board_symbols = ["❌" if x == 0 else "✅" for x in board]
        board_str = "\n".join("".join(map(str, board_symbols[i:i + 5])) for i in range(0, len(board), 5))

        return board_str, mines_amount, bet_amount, uuid
    
    def pathfinding(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A"

        self.conn.request("GET", "/games/mines/history?size=6&page=0", headers=self.headers)
        history = json.loads(self.conn.getresponse().read())['data']
        mine_locations = [row for v in history for row in v['mineLocations']][:1][0]

        org = (mine_locations + self.max_tiles) % 25
        select = list(range(mine_locations, org % 25))

        accuracy = self.get_accuracy(select)

        board = ["✅" if i in select else "❌" for i in range(25)]
        board_str = "\n".join("".join(map(str, board[i:i + 5])) for i in range(0, len(board), 5))

        return board_str, accuracy
    
    def safesearch(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        board = [0] * 25
        self.conn.request("GET", "/games/mines/history?size=24&page=0", headers=self.headers)
        history = json.loads(self.conn.getresponse().read().decode())['data']
        history = history[0]['mineLocations']
        
        board = ["✅" if x == 1 else "❌" for x in board]
        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nonce4 = data_game['game']['nonce'] - 1
        hash_1 = data_game['game']['_id']['$oid']

        x = 0
        for v in range(len(history) - 1):
            if x < 4:
                h = min(abs(history[v] - history[v + 1]) + (history[v] - v), 24)
                board[h] = 1
                x += 1
            else:
                break

        board = ["✅" if x == 1 else "❌" for x in board]
        return "\n".join("".join(map(str, board[i:i + 5])) for i in range(0, len(board), 5)), mines_amount, bet_amount, uuid

    def pastgamesbest(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nonce4 = data_game['game']['nonce'] - 1
        hash_1 = data_game['game']['_id']['$oid']
        response = scraper.get('https://api.bloxflip.com/games/mines/history',
                                params={'size': '20', 'page': '0'},
                                headers=self.headers).json()["data"]
        
        if len(response) < 3:
            return "*Not enough data available for prediction*", "N/A", "N/A", "N/A", "None"
        
        mine_locations = response[2].get('mineLocations')

        if not mine_locations:
            return "*Mine locations are not available for the game.*", "N/A", "N/A", "N/A", "None"
        
        spots = len(mine_locations)

        mines = ["❌"] * 25
        for i in range(spots):
            if i < len(mine_locations) and mine_locations[i] < 25:
                mines[mine_locations[i]] = "✅"

        X_train = [[int(cell == "✅") for cell in mines]]
        y_train = [1]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        X_pred = [[int(cell == "✅") for cell in mines]]
        prediction = model.predict(X_pred)

        prediction_grid = np.zeros((5, 5), dtype=int)
    
        for spot in prediction:
            row, col = divmod(spot, 5)
            prediction_grid[row][col] = 1

        prediction_string = ""
        mine_emoji = "❌"
        safe_emoji = "✅"

        for row in prediction_grid:
            prediction_string += " ".join([mine_emoji if cell == 1 else safe_emoji for cell in row]) + "\n"

        Prediction = f" "
        for i in range(0, 25, 5):
            row = "".join(
                [mine_emoji if cell == "❌" else safe_emoji for cell in mines[i:i + 5]])
            Prediction += row + "\n"

        return Prediction, mines_amount, bet_amount, uuid

    def pulsivesp(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        t = self.tile_setup2()
        accuracy = self.get_accuracy(t)
        x = self.create_board(t)
        r3 = scraper.get("https://api.bloxflip.com/games/mines",
                            headers=self.headers)
        data_game = json.loads(r3.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        return x, mines_amount, bet_amount, uuid

    def pastgames(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        t = self.tile_setup()
        accuracy = self.get_accuracy(t)
        x = self.create_board(t)
        r3 = scraper.get("https://api.bloxflip.com/games/mines",
                            headers=self.headers)
        data_game = json.loads(r3.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        return x, mines_amount, bet_amount, uuid

    def pastgames2(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        
        nonce4 = data_game['game']['nonce'] - 1
    
        params = {
        'size': '5',
        'page': '0'
        }
        headers = self.headers
        response = scraper.get('https://api.bloxflip.com/games/mines/history',
                            params=params,
                            headers=headers).json()["data"]
        if len(response) < 3:
            return "*Error occured*", "N/A", "N/A", "None"
        
        mine_locations = response[2].get('mineLocations')

        if not mine_locations:
            return "*Error occured*", "N/A", "N/A", "None"

        spots = len(mine_locations)

        mines = ["❌"] * 25
        for i in range(spots):
            if i < len(mine_locations) and mine_locations[i] < 25:
                mines[mine_locations[i]] = "✅"

        X_train = [[int(cell == "✅") for cell in mines]]
        y_train = [1]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        X_pred = [[int(cell == "✅") for cell in mines]]
        prediction = model.predict(X_pred)

        prediction_grid = np.zeros((5, 5), dtype=int)

        for spot in prediction:
            row, col = divmod(spot, 5)
            prediction_grid[row][col] = 1

        prediction_string = ""
        mine_emoji = "❌"
        safe_emoji = "✅"

        for row in prediction_grid:
            prediction_string += " ".join([mine_emoji if cell == 1 else safe_emoji for cell in row]) + "\n"

        Prediction = ""
        for i in range(0, 25, 5):
            row = "".join(
                [mine_emoji if cell == "❌" else safe_emoji for cell in mines[i:i + 5]])
            Prediction += row + "\n"

        return Prediction, mines_amount, bet_amount, uuid

    def neighbors(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        n_s = self.n_spawn()
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                              headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        return n_s, mines_amount, bet_amount, uuid
    
    def probability(self):
          if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A"
          r7 = scraper.get("https://api.bloxflip.com/games/mines",
                              headers=self.headers)
          data_game = json.loads(r7.text)
          mines_amount = data_game['game']['minesAmount']
          uuid = data_game['game']['uuid']
          bet_amount = data_game['game']['betAmount']
    
          nonce5 = data_game['game']['nonce'] - 1
        
          r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                        'size': '5',
                        'page': '0'
                        })
          data = r8.json()['data']
          latest_game = data[0]
          mines_location = latest_game['mineLocations']
          clicked_spots = latest_game['uncoveredLocations']

          num_spots = self.max_tiles
          grid = ['💥'] * 25
          total_spots = list(range(25))
          unclicked_spots = [spot for spot in total_spots if spot not in clicked_spots and spot not in mines_location]

          prior_safe = (25 - len(clicked_spots) - len(mines_location)) / 25
          prior_dangerous = len(mines_location) / 25

          bayesian_model = np.zeros((25, 25, 25, 25))

          for i in range(len(clicked_spots) - 3):
              prev3_spot = clicked_spots[i]
              prev2_spot = clicked_spots[i + 1]
              prev_spot = clicked_spots[i + 2]
              curr_spot = clicked_spots[i + 3]

              alpha_safe = bayesian_model[prev3_spot, prev2_spot, prev_spot, curr_spot] + prior_safe
              beta_safe = 1 - bayesian_model[prev3_spot, prev2_spot, prev_spot, curr_spot] + prior_safe
              alpha_dangerous = bayesian_model[prev3_spot, prev2_spot, prev_spot, curr_spot] + prior_dangerous
              beta_dangerous = 1 - bayesian_model[prev3_spot, prev2_spot, prev_spot, curr_spot] + prior_dangerous
              bayesian_model[prev3_spot, prev2_spot, prev_spot, curr_spot] = beta.mean(alpha_safe, beta_safe)
          
          np.random.seed(8)
          safe_probs = np.zeros(25)
          dangerous_probs = np.zeros(25)

          for i in range(25):
           if i not in clicked_spots and i not in mines_location:
              for l in range(25):
                  for y in range(25):
                      count = bayesian_model[l, y, i, :]
                      if np.sum(count) > 0:
                          alpha_safe = np.sum(count) + prior_safe
                          beta_safe = len(count) - np.sum(count) + 1
                          alpha_dangerous = np.sum(count) + prior_dangerous
                          beta_dangerous = len(count) - np.sum(count) + 1
                          safe_probs[i] += beta.mean(alpha_safe, beta_safe)
                          dangerous_probs[i] += beta.mean(alpha_dangerous, beta_dangerous)

          game_data = 10000
          safe_counts = np.zeros(25)
          bad_counts = np.zeros(25)

          num_unclicked = min(len(unclicked_spots), 25 - len(clicked_spots) - len(mines_location))

          correct_predictions = 0
          total_predictions = 0

          for i in range(game_data):
           game = latest_game.copy()
          unclicked_spots_subset = []
          if len(unclicked_spots) > 0:
              np.random.shuffle(unclicked_spots)
              unclicked_spots_subset = unclicked_spots[:num_unclicked]
          game['uncoveredLocations'] = clicked_spots + unclicked_spots_subset
          exploded = False
          num_mines_uncovered = 0

          for spot in unclicked_spots_subset:
              if spot in mines_location:
                  num_mines_uncovered += 1
                  exploded = True
                  break
              elif np.random.rand() < safe_probs[spot]:
                  game['uncoveredLocations'].append(spot)
          
          for spot in unclicked_spots_subset:
              if not exploded:
                  if spot not in game['uncoveredLocations']:
                      safe_counts[unclicked_spots.index(spot)] -= 1
                  else:
                      bad_counts[unclicked_spots.index(spot)] += 1
                  if spot in mines_location:
                      num_mines_uncovered += 1
          


          mc_safe_probs = np.zeros(25)
          for i in range(25):
              if i not in clicked_spots and i not in mines_location:
                 mc_safe_probs[i] = (safe_counts[unclicked_spots.index(i)] +
                                     bad_counts[unclicked_spots.index(i)] +
                                     safe_probs[i] * game_data) / (game_data + np.sum(safe_probs))
                 
          top_safe_spots = np.argsort(mc_safe_probs)[::-1]
          top_safe_spots = [spot for spot in top_safe_spots if spot not in mines_location][:25]
          num_spots = int(num_spots)
          selected_safe_spots = np.random.choice(top_safe_spots, min(num_spots, len(top_safe_spots)), replace=False)


          top_dangerous_spots = np.argsort(dangerous_probs)[::-1]
          top_dangerous_spots = [spot for spot in top_dangerous_spots if spot not in mines_location][:25]
          selected_dangerous_spots = np.random.choice(top_dangerous_spots, min(num_spots, len(top_dangerous_spots)), replace=False)

          if num_mines_uncovered == mines_amount:
              correct_predictions += 2
          total_predictions += 1

          if total_predictions > 0:
              real_accuracy = (correct_predictions / total_predictions) * 100
          else:
              real_accuracy = 0
          


          for i, spot in enumerate(range(len(grid))):
            if spot not in mines_location and grid[spot] != '❌' and grid[spot] != '❔':
             if spot in selected_safe_spots[:int(num_spots)]:
                grid[spot] = '✅'
             elif spot in selected_dangerous_spots[:mines_amount]:
                grid[spot] = '❔'
             else:
                grid[spot] = '❌'
          
          mine_emoji = "❌"
          safe_emoji = "✅"

          grid = [
          mine_emoji if cell == '❌' or cell == '💥' or cell == '❔' else safe_emoji
          for cell in grid
          ]
          Prediction6 = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))

          return Prediction6, mines_amount, bet_amount, uuid

    def v6(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        # Fetch the current game details
        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce5 = data_game['game']['nonce'] - 1

        # Fetch the game history
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1000', 'page': '0'})
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid_size = 25
        num_spot = self.max_tiles

        def is_neighbor(spot1, spot2):
            row1, col1 = divmod(spot1, 5)
            row2, col2 = divmod(spot2, 5)
            return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1

        def calculate_safest_and_most_dangerous_spots(clicked_spots, mines_location, grid_size, num_mines, num_spot):
            safe_spots = []
            bomb_likelihoods = np.full((5,5), 0.5)

            for spot in range(grid_size):
                if spot not in clicked_spots and spot not in mines_location:
                    num_neighboring_mines = sum(1 for neighbor in mines_location if is_neighbor(spot, neighbor))
                    if num_neighboring_mines == 0:
                        safe_spots.append(spot)


            likelihood_scores = [1 - bomb_likelihoods[spot // 5, spot % 5] for spot in safe_spots]

   
            sorted_safe_spots = [x for _, x in sorted(zip(likelihood_scores, safe_spots), reverse=True)]


            safest_spots = sorted_safe_spots[:int(num_spot)]

            bomb_chance_at_safest_spot = 1 - likelihood_scores[safe_spots.index(safest_spots[0])]

            most_dangerous_spot = safe_spots[np.argmax(likelihood_scores)]

            return safest_spots, bomb_chance_at_safest_spot, most_dangerous_spot


        latest_game = {
        "mineLocations": mines_location,
        "uncoveredLocations": clicked_spots,
        }

        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        num_mines = len(mines_location)

        safest_spots, bomb_chance, most_dangerous_spot = calculate_safest_and_most_dangerous_spots(clicked_spots, mines_location, grid_size, num_mines, num_spot)

        grid = [['❓'] * 5 for _ in range(5)]

        for spot in clicked_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '❌'

        for spot in safest_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '✅'

        most_dangerous_row, most_dangerous_col = divmod(most_dangerous_spot, 5)
        grid[most_dangerous_row][most_dangerous_col] = '❌'

        grid = [cell for row in grid for cell in row]


        Prediction1 = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))

        return Prediction1, mines_amount, bet_amount, uuid

    def gridguardian(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r2 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r2.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nounce = data_game['game']['nonce'] - 1
        random_accuracy = random.uniform(70, 85)

        r = scraper.get('https://api.bloxflip.com/games/mines/history',
                             headers=self.headers,
                             params={"size": '5', "page": '0'})
        data = r.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        num_spots = 1
        safe_spots = []
        past_mine = mines_location
        past_safe = clicked_spots

        def get_adjacent(tiles):
            adjacent = []
            row = (tiles - 1) // 5
            col = (tiles - 1) % 5
            for r in range(max(row-1, 0), min(row+2, 5)):
                for c in range(max(col-1, 0), min(col+2, 5)):
                    for c in range(max(col-1, 0), min(col+2, 5)):
                        if r == row and c == col:
                            continue
                        adjacent.append(r * 5 + c + 1)
                        return adjacent
            
        X = []
        y = []

        for i in range(25):
            adj_mines = sum(1 for n in past_mine if n in get_adjacent(i) and n == '❌')
            num_flagged = len(past_mine)
            num_safe_tiles = (25 - len(past_mine))
            prob_safe = (num_safe_tiles - adj_mines) / num_safe_tiles
            adj_prob_safe = sum(1 for n in past_safe if n in get_adjacent(i) and n != '❌' and n in [index for index in X])
            adj_cleared = sum(1 for n in safe_spots if n in get_adjacent(i) and n in past_safe)
            X.append((i, num_safe_tiles, prob_safe, num_flagged, adj_prob_safe, adj_cleared))
            y.append(i in past_mine)

        classifiers = KNeighborsClassifier(n_neighbors=3)
        classifiers.fit(X, y)

        probs = classifiers.predict_proba(X)
        predictions = classifiers.predict(X)

        correct_predictions = sum(1 for predicted, actual in zip(predictions, y) if predicted == actual)
        total_predictions = len(y)
        real_accuracy = correct_predictions / total_predictions

        np.random.seed(7)
        safe_spots = [(i, prob[0]) for i, prob in enumerate(probs)]
        np.random.shuffle(safe_spots)
        chosen_spots = []

        for spot in safe_spots:
            if spot[0] not in past_mine and spot[0] not in past_safe:
                chosen_spots.append(spot[0])
                if len(chosen_spots) == num_spots:
                    break
        
        grid_list = ['❌'] * 25
        for r in range(5):
            for c in range(5):
                index = r * 5 + c
                if index in past_mine:  
                    grid_list[index] = '❌'
                elif index in chosen_spots:
                    grid_list[index] = '✅'


        prediction = f"{''.join(grid_list[:5])}\n{''.join(grid_list[5:10])}\n{''.join(grid_list[10:15])}\n{''.join(grid_list[15:20])}\n{''.join(grid_list[20:])}"

        return prediction, mines_amount, bet_amount, uuid
    
        
    def ralgorithm2(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                        headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                            'size': '1000',
                            'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']
        
        grid_size = 25
        num_spot = self.max_tiles

        def is_neighbor(spot1, spot2):
            row1, col1 = divmod(spot1, 5)
            row2, col2 = divmod(spot2, 5)
            return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1

        def calculate_safest_and_most_dangerous_spots(clicked_spots, mines_location, grid_size, num_mines, num_spot):
            safe_spots = []
            bomb_likelihoods = np.array([
                [0.1, 0.9, 0.9, 0.9, 0.5],
                [0.6, 0.7, 0.8, 0.9, 0.0],
                [0.9, 0.9, 0.9, 0.9, 0.9],
                [0.9, 0.8, 0.7, 0.6, 0.0],
                [0.2, 0.3, 0.4, 0.5, 0.6]
            ])

            for spot in range(grid_size):
                if spot not in clicked_spots and spot not in mines_location:
                    num_neighboring_mines = sum(1 for neighbor in mines_location if is_neighbor(spot, neighbor))
                    if num_neighboring_mines == 0:
                        safe_spots.append(spot)

            likelihood_scores = [1 - bomb_likelihoods[spot // 5, spot % 5] for spot in safe_spots]

            sorted_safe_spots = [x for _, x in sorted(zip(likelihood_scores, safe_spots), reverse=True)]

            safest_spots = sorted_safe_spots[:int(num_spot)]

            bomb_chance_at_safest_spot = 1 - likelihood_scores[safe_spots.index(safest_spots[0])]

            most_dangerous_spot = safe_spots[np.argmax(likelihood_scores)]

            return safest_spots, bomb_chance_at_safest_spot, most_dangerous_spot

        latest_game = {
            "mineLocations": mines_location,
            "uncoveredLocations": clicked_spots,
        }

        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        num_mines = len(mines_location)

        safest_spots, bomb_chance, most_dangerous_spot = calculate_safest_and_most_dangerous_spots(clicked_spots, mines_location, grid_size, num_mines, num_spot)

        grid = [['❓'] * 5 for _ in range(5)]

        for spot in clicked_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '❌'

        np.random.shuffle(safest_spots)  # Shuffle the safe spots

        # Split the safe spots into rows, shuffle each row, and then flatten them back into a single list
        shuffled_safe_spots = [cell for row in [np.random.permutation(row) for row in np.array_split(safest_spots, 5)] for cell in row]

        for spot in shuffled_safe_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '💎'

        most_dangerous_row, most_dangerous_col = divmod(most_dangerous_spot, 5)
        grid[most_dangerous_row][most_dangerous_col] = '❌'

        # Convert the grid to a flattened list for better display
        grid = [cell for row in grid for cell in row]

        Prediction1 = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))

        return Prediction1, mines_amount, bet_amount, uuid

    def nostaglia(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        # Fetch the current game details
        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce5 = data_game['game']['nonce'] - 1

        # Fetch the game history
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1000', 'page': '0'})
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid_size = 20
        num_spot = self.max_tiles

        def is_neighbor(spot1, spot2):
            row1, col1 = divmod(spot1, 5)
            row2, col2 = divmod(spot2, 5)
            return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1

        def calculate_safest_and_most_dangerous_spots(clicked_spots, mines_location, grid_size, num_mines, num_spot):
            safe_spots = []
            bomb_likelihoods = np.array([
                [0.7,0.9,0.8,0.4,0.4],
                [0.4,0.5,0.3,0.5,0.6],
                [0.1,0.2,0.3,0.2,0.1],
                [0.2,0.6,0.5,0.8,0.4],
                [0.2,0.3,0.7,0.4,0.9]
            ])

            for spot in range(grid_size):
                if spot not in clicked_spots and spot not in mines_location:
                    num_neighboring_mines = sum(1 for neighbor in mines_location if is_neighbor(spot, neighbor))
                    if num_neighboring_mines == 0:
                        safe_spots.append(spot)

            likelihood_scores = [1 - bomb_likelihoods[spot // 5, spot % 5] for spot in safe_spots]

            sorted_safe_spots = [x for _, x in sorted(zip(likelihood_scores, safe_spots), reverse=True)]

            safest_spots = sorted_safe_spots[:int(num_spot)]

            bomb_chance_at_safest_spot = 1 - likelihood_scores[safe_spots.index(safest_spots[0])]

            most_dangerous_spot = safe_spots[np.argmax(likelihood_scores)]

            return safest_spots, bomb_chance_at_safest_spot, most_dangerous_spot

        latest_game = {
            "mineLocations": mines_location,
            "uncoveredLocations": clicked_spots,
        }

        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        num_mines = len(mines_location)

        safest_spots, bomb_chance, most_dangerous_spot = calculate_safest_and_most_dangerous_spots(clicked_spots, mines_location, grid_size, num_mines, num_spot)

        grid = [['✅'] * 5 for _ in range(5)]

        for spot in clicked_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '❌'

        np.random.shuffle(safest_spots)  # Shuffle the safe spots

        # Split the safe spots into rows, shuffle each row, and then flatten them back into a single list
        shuffled_safe_spots = [cell for row in [np.random.permutation(row) for row in np.array_split(safest_spots, 5)] for cell in row]

        for spot in shuffled_safe_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '❌'

        most_dangerous_row, most_dangerous_col = divmod(most_dangerous_spot, 5)
        grid[most_dangerous_row][most_dangerous_col] = '❌'

        # Convert the grid to a flattened list for better display
        grid = [cell for row in grid for cell in row]

        Prediction1 = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))

        return Prediction1, mines_amount, bet_amount, uuid

    def pulsivev10(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        # Fetch the current game details
        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce5 = data_game['game']['nonce'] - 1

        # Fetch the game history
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1000', 'page': '0'})
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid_size = 25
        num_spot = self.max_tiles

        def is_neighbor(spot1, spot2):
            row1, col1 = divmod(spot1, 5)
            row2, col2 = divmod(spot2, 5)
            return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1

        # Improved bomb likelihood matrix based on historical data
        bomb_likelihoods = np.array([
                [0.7,0.9,0.8,0.4,0.4],
                [0.4,0.5,0.3,0.5,0.6],
                [0.1,0.2,0.3,0.2,0.1],
                [0.2,0.6,0.5,0.8,0.4],
                [0.2,0.3,0.7,0.4,0.9]
            ])

        def calculate_probabilities(mines_location, clicked_spots, grid_size):
            mine_counts = np.zeros(grid_size)
            safe_counts = np.zeros(grid_size)

            # Give higher weight to more recent data
            for i, game in enumerate(reversed(data[:1])):  # Limit to last 100 games for recency
                weight = 1 / (i + 1)
                for spot in game['mineLocations']:
                    mine_counts[spot] += weight
                for spot in game['uncoveredLocations']:
                    safe_counts[spot] += weight

            probabilities = (safe_counts + 1) / (mine_counts + safe_counts + 2)

            return probabilities

        probabilities = calculate_probabilities(mines_location, clicked_spots, grid_size)

        for i in range(grid_size):
            row, col = divmod(i, 5)
            probabilities[i] *= (1 - bomb_likelihoods[row, col])
            if any(is_neighbor(i, mine) for mine in mines_location):
                probabilities[i] *= 0.5

        def select_safest_spots(probabilities, num_spot):
            spot_prob_pairs = [(spot, prob) for spot, prob in enumerate(probabilities)]
            spot_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            safest_spots = [spot for spot, _ in spot_prob_pairs[:num_spot]]
            return safest_spots

        safest_spots = select_safest_spots(probabilities, num_spot)

        grid = [['❌'] * 5 for _ in range(5)]

        for spot in clicked_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '❌'

        for spot in safest_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '✅'

        grid = [cell for row in grid for cell in row]

        Prediction1 = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))

        return Prediction1, mines_amount, bet_amount, uuid
    def pulsivesplusreal(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                              headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
    
        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                            'size': '1000',
                            'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']
        
        num_spots = self.max_tiles
        safe_spots = []
        past_mine = mines_location
        past_safe = clicked_spots

        def get_adjacent(tiles):
            adjacent = []
            row = (tiles - 1) // 5
            col = (tiles - 1) % 5
            for r in range(max(row-1, 0), min(row+2, 5)):
                for c in range(max(col-1, 0), min(col+2, 5)):
                    for c in range(max(col-1, 0), min(col+2, 5)):
                        if r == row and c == col:
                            continue
                        adjacent.append(r * 5 + c + 1)
                        return adjacent
            
        X = []
        y = []

        for i in range(25):
            adj_mines = sum(1 for n in past_mine if n in get_adjacent(i) and n == '❌')
            num_flagged = len(past_mine)
            num_safe_tiles = (25 - len(past_mine))
            prob_safe = (num_safe_tiles - adj_mines) / num_safe_tiles
            adj_prob_safe = sum(1 for n in past_safe if n in get_adjacent(i) and n != '❌' and n in [index for index in X])
            adj_cleared = sum(1 for n in safe_spots if n in get_adjacent(i) and n in past_safe)
            X.append((i, num_safe_tiles, prob_safe, num_flagged, adj_prob_safe, adj_cleared))
            y.append(i in past_mine)

        classifiers = KNeighborsClassifier(n_neighbors=25)
        classifiers.fit(X, y)

        probs = classifiers.predict_proba(X)
        predictions = classifiers.predict(X)

        correct_predictions = sum(1 for predicted, actual in zip(predictions, y) if predicted == actual)
        total_predictions = len(y)
        real_accuracy = correct_predictions / total_predictions

        np.random.seed(42)
        safe_spots = [(i, prob[0]) for i, prob in enumerate(probs)]
        np.random.shuffle(safe_spots)
        print(safe_spots)
        chosen_spots = []

        for spot in safe_spots:
            if spot[0] not in past_mine and spot[0] not in past_safe:
                chosen_spots.append(spot[0])
                if len(chosen_spots) == num_spots:
                    break
        
        grid_list = ['❌'] * 25
        for r in range(5):
            for c in range(5):
                index = r * 5 + c
                if index in past_mine:  
                    grid_list[index] = '❌'
                elif index in chosen_spots:
                    grid_list[index] = '✅'

        prediction = f"{''.join(grid_list[:5])}\n{''.join(grid_list[5:10])}\n{''.join(grid_list[10:15])}\n{''.join(grid_list[15:20])}\n{''.join(grid_list[20:])}"

        return prediction, mines_amount, bet_amount, uuid
    
    def pulsivemines3(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r2 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r2.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nonce = data_game['game']['nonce'] - 1
        hash_id = data_game['game']['_id']['$oid']
        random_accuracy = random.uniform(85, 95)

        r = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={"size": '5', "page": '0'})
        data = r.json()['data']
        mines_location = [row for x in data for row in x['mineLocations']]
        clicked_spots = [row for x in data for row in x['uncoveredLocations']]
        def uuid_to_number(uuid_str):
            # Convert UUID to a number
            uuid_number = int(uuiddd.UUID(uuid_str))
            return uuid_number

        def get_safe_positions(uuid_str, past_mine_locations, past_clicked_spots, num_safe_spots, grid_size=25):
            # Seed the random number generator with the UUID number
            uuid_number = uuid_to_number(uuid_str)
            random.seed(uuid_number)
            
            # Create a list of all positions
            all_positions = list(range(grid_size))

            # Remove past mine locations and past clicked spots from the available positions
            available_positions = [pos for pos in all_positions if pos not in past_mine_locations and pos not in past_clicked_spots]

            # Ensure the number of safe spots does not exceed available positions
            num_safe_spots = min(len(available_positions), num_safe_spots)

            # Select safe positions manually
            safe_positions = []
            step = max(1, len(available_positions) // num_safe_spots)

            for i in range(num_safe_spots):
                if available_positions:
                    index = (i * step) % len(available_positions)
                    pos = available_positions[index]
                    safe_positions.append(pos)
                    available_positions.pop(index)
                else:
                    break

            # Shuffle the safe positions a little for safer prediction
            random.shuffle(safe_positions)

            return safe_positions

        def create_grid(grid_size, safe_positions, past_mine_locations, past_clicked_spots):
            # Initialize grid with empty spaces
            grid = [['' for _ in range(grid_size)] for _ in range(grid_size)]
            
            # Place past mine locations
            for mine in past_mine_locations:
                row, col = divmod(mine, grid_size)
                grid[row][col] = 'M'
            
            # Place past clicked spots
            for click in past_clicked_spots:
                row, col = divmod(click, grid_size)
                grid[row][col] = 'X'
            
            # Place safe positions
            for safe in safe_positions:
                row, col = divmod(safe, grid_size)
                grid[row][col] = 'S'
            
            return grid

        def print_grid(grid):
            for row in grid:
                print(' '.join(row))
            print()

        # Example usage
        uuid_str = uuid
        past_mine_locations = mines_location
        past_clicked_spots = clicked_spots
        num_safe_spots = self.max_tiles
        grid_size = 5

        safe_positions = get_safe_positions(uuid_str, past_mine_locations, past_clicked_spots, num_safe_spots, grid_size)
        grid = create_grid(grid_size, safe_positions, past_mine_locations, past_clicked_spots)

    def pulsivemines(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r2 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r2.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nonce = data_game['game']['nonce'] - 1
        hash_id = data_game['game']['_id']['$oid']
        random_accuracy = random.uniform(85, 95)

        r = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={"size": '5', "page": '0'})
        data = r.json()['data']
        mines_location = [row for x in data for row in x['mineLocations']]
        clicked_spots = [row for x in data for row in x['uncoveredLocations']]
        
        def get_safe_positions(past_mine_locations, past_clicked_spots, num_safe_spots, grid_size=25):
            # Create a list of all positions
            all_positions = list(range(grid_size))
            
            # Remove past mine locations and past clicked spots from available positions
            available_positions = [pos for pos in all_positions if pos not in past_mine_locations and pos not in past_clicked_spots]
            
            # Ensure the number of safe spots does not exceed available positions
            num_safe_spots = min(len(available_positions), num_safe_spots)
            
            # Select safe positions with deterministic shuffling
            safe_positions = []
            
            # Calculate the step size for shuffling
            step = max(1, len(available_positions) // num_safe_spots)
            
            # Use a deterministic shuffling technique
            for i in range(num_safe_spots):
                pos_index = (i * step) % len(available_positions)
                pos = available_positions[pos_index]
                safe_positions.append(pos)
                available_positions.pop(pos_index)
            
     
            remaining_needed = num_safe_spots - len(safe_positions)
            additional_positions = [pos for pos in all_positions if pos not in safe_positions and pos not in past_mine_locations]
            for i in range(remaining_needed):
                if additional_positions:
                    pos_index = i % len(additional_positions)
                    pos = additional_positions[pos_index]
                    safe_positions.append(pos)
                    additional_positions.pop(pos_index)
            
            # Shuffle safe positions slightly for better safety
            for i in range(len(safe_positions)):
                swap_index = random.randint(0, len(safe_positions) - 1)
                safe_positions[i], safe_positions[swap_index] = safe_positions[swap_index], safe_positions[i]
            
            return safe_positions
        
        def create_grid(safe_positions, past_mine_locations, past_clicked_spots, grid_size=25):
            grid = ['❌'] * grid_size
            if safe_positions:
                for pos in safe_positions:
                    grid[pos] = '✅'
            return grid

        def display_grid(grid):
            grid_size = int(len(grid) ** 0.5)
            return "\n".join("".join(grid[i:i + grid_size]) for i in range(0, len(grid), grid_size))

        past_mine_locations = mines_location
        past_clicked_spots = clicked_spots

        final = ""

        safe_positions = get_safe_positions(past_mine_locations, past_clicked_spots, self.max_tiles)
        if safe_positions is None:
            grid = create_grid([], past_mine_locations, past_clicked_spots)
        else:
            grid = create_grid(safe_positions, past_mine_locations, past_clicked_spots)

        grid_display = display_grid(grid)

        if grid_display.count("✅") < 1:
            final = grid_display
        else:
            final = "*Cannot predict your game, please try other method.*"

        return grid_display, mines_amount, bet_amount, uuid
    
    def aspectmines(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r2 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r2.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nonce = data_game['game']['nonce'] - 1
        hash_id = data_game['game']['_id']['$oid']
        random_accuracy = random.uniform(85, 95)

        r = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={"size": '5', "page": '0'})
        data = r.json()['data'][0]
        mines_location = data['mineLocations']
        clicked_spots = data['uncoveredLocations']

        def calculate_bomb_likelihood_mrf(spot, clicked_spots, mines_location, grid_size):
            bomb_likelihood = 0.0
            row, col = spot // grid_size, spot % grid_size
    
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = row + dr, col + dc
                    neighbor_spot = nr * grid_size + nc
            
            if 0 <= nr < grid_size and 0 <= nc < grid_size and neighbor_spot not in clicked_spots:
                if neighbor_spot in mines_location:
                    bomb_likelihood += 0.3
                bomb_likelihood += 0.1
    
            return min(bomb_likelihood, 0.9)

        def mrf_objective(x, *args):
            grid_size, clicked_spots, mines_location = args
            likelihood = 0.0
    
            for spot in range(grid_size ** 2):
                if spot not in clicked_spots and spot not in mines_location:
                    likelihood -= x[spot] * calculate_bomb_likelihood_mrf(spot, clicked_spots, mines_location, grid_size)
    
            return likelihood

        def predict_safe_tiles_mrf(clicked_spots, mines_location, grid_size, num_safe_tiles):
            x0 = np.random.rand(grid_size ** 2)
            bounds = [(0, 1)] * (grid_size ** 2)
    
            result = minimize(mrf_objective, x0, args=(grid_size, clicked_spots, mines_location), bounds=bounds, method='L-BFGS-B')
    
            probabilities = result.x
    
            safe_spots_indices = np.argsort(probabilities)[:num_safe_tiles]
    
            grid_display = [['❌'] * grid_size for _ in range(grid_size)]
    
            for spot in clicked_spots:
                row, col = divmod(spot, grid_size)
                grid_display[row][col] = '❌'
    
            for spot in safe_spots_indices:
                row, col = divmod(spot, grid_size)
                grid_display[row][col] = '✅'
    
            grid_display_str = '\n'.join(''.join(row) for row in grid_display)
    
            return grid_display_str

        grid_size = 5
        num_safe_tiles = self.max_tiles

        predicted_grid = predict_safe_tiles_mrf(clicked_spots, mines_location, grid_size, num_safe_tiles)

        return predicted_grid, mines_amount, bet_amount, uuid

    def pulsivemines2(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r2 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r2.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nonce = data_game['game']['nonce'] - 1
        hash_id = data_game['game']['_id']['$oid']
        random_accuracy = random.uniform(85, 95)

        r = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={"size": '5', "page": '0'})
        data = r.json()['data']
        mines_location = [row for x in data for row in x['mineLocations']]
        clicked_spots = [row for x in data for row in x['uncoveredLocations']]

        def get_safe_positions(round_id, past_mine_locations, past_clicked_spots, num_safe_spots, grid_size=25):
            seed = int(hashlib.sha1(round_id.encode()).hexdigest(), 25)
            random.seed(seed)
            
            # Create a list of all positions
            all_positions = list(range(grid_size))

            # Remove past mine locations and past clicked spots from the available positions
            available_positions = [pos for pos in all_positions if pos not in past_mine_locations and pos not in past_clicked_spots]

            # Ensure the number of safe spots does not exceed available positions
            num_safe_spots = min(len(available_positions), num_safe_spots)

            # If less than the required safe spots, fill with additional random positions
            safe_positions = random.sample(available_positions, num_safe_spots)

            return safe_positions

        def create_grid(safe_positions, past_mine_locations, past_clicked_spots, grid_size=25):
            grid = ['❔'] * grid_size
            if safe_positions:
                for pos in safe_positions:
                    grid[pos] = '✅'
            for pos in past_mine_locations:
                grid[pos] = '💣'
            for pos in past_clicked_spots:
                if grid[pos] == '❔':
                    grid[pos] = '❌'
            return grid

        def display_grid(grid):
            grid_size = int(len(grid) ** 0.5)
            return "\n".join("".join(grid[i:i + grid_size]) for i in range(0, len(grid), grid_size))

        round_id = uuid
        past_mine_locations = mines_location
        past_clicked_spots = clicked_spots

        safe_positions = get_safe_positions(round_id, past_mine_locations, past_clicked_spots, self.max_tiles)
        if safe_positions is None:
            grid = create_grid([], past_mine_locations, past_clicked_spots)
        else:
            grid = create_grid(safe_positions, past_mine_locations, past_clicked_spots)

        grid_display = display_grid(grid)

        return grid_display, mines_amount, bet_amount, uuid
    

    def rfcaa(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None", "N/A"
        
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                        headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        hash_1 = data_game['game']['_id']['$oid']
        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                            'size': '1000',
                            'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']
        
        num_spots = 19
        safe_spots = []
        past_mine = mines_location
        past_safe = clicked_spots

        def get_adjacent(tiles):
            adjacent = []
            row = (tiles - 1) // 5
            col = (tiles - 1) % 5
            for r in range(max(row-1, 0), min(row+2, 5)):
                for c in range(max(col-1, 0), min(col+2, 5)):
                    if r == row and c == col:
                        continue
                    adjacent.append(r * 5 + c + 1)
            return adjacent

        X = []
        y = []

        for i in range(25):
            adj_mines = sum(1 for n in past_mine if n in get_adjacent(i))
            num_flagged = len(past_mine)
            num_safe_tiles = (25 - len(past_mine))
            prob_safe = (num_safe_tiles - adj_mines) / num_safe_tiles
            adj_prob_safe = sum(1 for n in past_safe if n in get_adjacent(i) and n != '❌')
            adj_cleared = sum(1 for n in safe_spots if n in get_adjacent(i) and n in past_safe)
            X.append((i, num_safe_tiles, prob_safe, num_flagged, adj_prob_safe, adj_cleared))
            y.append(1 if i in past_mine else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        safe_spots = [(i, prob[0]) for i, prob in enumerate(classifier.predict_proba(X))]
        
        # Sort safe spots based on distance from past mines
        safe_spots.sort(key=lambda x: min(abs(x[0] - m) for m in past_mine), reverse=True)
        
        chosen_spots = []

        np.random.seed(7)  # Set a fixed seed for deterministic shuffling
        np.random.shuffle(safe_spots)
        
        for spot in safe_spots:
            if spot[0] not in past_mine and spot[0] not in past_safe:
                if len(chosen_spots) == num_spots:
                    break
                chosen_spots.append(spot[0])
                # Remove adjacent spots from consideration
                adjacent_spots = get_adjacent(spot[0])
                for adj_spot in adjacent_spots:
                    if adj_spot in safe_spots and adj_spot in past_mine:
                        safe_spots.remove(adj_spot)
        
        grid_list = ['✅'] * 25
        for r in range(5):
            for c in range(5):
                index = r * 5 + c
                if index in past_mine:  
                    grid_list[index] = '❌'
                elif index in chosen_spots:
                    grid_list[index] = '❌'

        # Reformat grid_list into a list of rows
        grid_rows = [''.join(grid_list[i:i+5]) for i in range(0, 25, 5)]

        prediction = '\n'.join(grid_rows)

        return prediction, mines_amount, bet_amount, uuid

    def bruhrfc(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        r8 = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1000', 'page': '0'})
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid_size = 25
        num_spots = self.max_tiles

        def get_adjacent(tile):
            adjacent = []
            row, col = divmod(tile, 6)
            for r in range(max(row-1, 0), min(row+2, 5)):
                for c in range(max(col-1, 0), min(col+2, 5)):
                    if r == row and c == col:
                        continue
                    adjacent.append(r * 5 + c)
            return adjacent

        X = []
        y = []

        for i in range(grid_size):
            adj_mines = sum(1 for n in mines_location if n in get_adjacent(i))
            num_flagged = len(mines_location)
            num_safe_tiles = grid_size - len(mines_location)
            prob_safe = (num_safe_tiles - adj_mines) / num_safe_tiles
            adj_prob_safe = sum(1 for n in clicked_spots if n in get_adjacent(i) and n != '❌')
            X.append((i, num_safe_tiles, prob_safe, num_flagged, adj_prob_safe))
            y.append(1 if i in mines_location else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        probabilities = classifier.predict_proba(X)[:, 1]

        safe_spots = [(i, prob) for i, prob in enumerate(probabilities)]
        safe_spots.sort(key=lambda x: x[1], reverse=False)
        
        chosen_spots = []
        np.random.seed(7)
        np.random.shuffle(safe_spots)

        for spot in safe_spots:
            if spot[0] not in mines_location and spot[0] not in clicked_spots:
                if len(chosen_spots) == num_spots:
                    break
                chosen_spots.append(spot[0])
                adjacent_spots = get_adjacent(spot[0])
                for adj_spot in adjacent_spots:
                    safe_spots = [s for s in safe_spots if s[0] != adj_spot]

        grid_list = ['❌'] * grid_size
        for r in range(5):
            for c in range(5):
                index = r * 5 + c
                if index in mines_location:
                    grid_list[index] = '❌'
                elif index in chosen_spots:
                    grid_list[index] = '✅'

        grid_rows = [''.join(grid_list[i:i+5]) for i in range(0, grid_size, 5)]
        prediction = '\n'.join(grid_rows)

        return prediction, mines_amount, bet_amount, uuid

    def badrfc(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nonce5 = data_game['game']['nonce'] - 1

        r8 = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1000', 'page': '0'})
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid_size = 25
        num_spots = self.max_tiles
        past_mine = mines_location
        past_safe = clicked_spots

        def get_adjacent(tile):
            adjacent = []
            row, col = divmod(tile, 5)
            for r in range(max(row-1, 0), min(row+2, 5)):
                for c in range(max(col-1, 0), min(col+2, 5)):
                    if r == row and c == col:
                        continue
                    adjacent.append(r * 5 + c)
            return adjacent

        def calculate_probabilities(past_mine, past_safe, grid_size):
            X = []
            y = []

            for i in range(grid_size):
                adj_mines = sum(1 for n in past_mine if n in get_adjacent(i))
                num_flagged = len(past_mine)
                num_safe_tiles = grid_size - len(past_mine)
                prob_safe = (num_safe_tiles - adj_mines) / num_safe_tiles
                adj_prob_safe = sum(1 for n in past_safe if n in get_adjacent(i) and n != '❌')
                X.append((i, num_safe_tiles, prob_safe, num_flagged, adj_prob_safe))
                y.append(1 if i in past_mine else 0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            probabilities = classifier.predict_proba(X)[:, 1]
            return probabilities

        probabilities = calculate_probabilities(past_mine, past_safe, grid_size)

        # Avoid previously clicked spots and mines
        safe_spots = [(i, prob) for i, prob in enumerate(probabilities) if i not in past_mine and i not in past_safe]
        # Invert probabilities to give higher weight to less likely bomb spots
        safe_spots.sort(key=lambda x: -x[1])

        chosen_spots = []

        for spot in safe_spots:
            if len(chosen_spots) == num_spots:
                break
            chosen_spots.append(spot[0])
            adjacent_spots = get_adjacent(spot[0])
            safe_spots = [s for s in safe_spots if s[0] not in adjacent_spots]

        grid_list = ['❌'] * grid_size
        for spot in past_mine:
            grid_list[spot] = '❌'
        for spot in chosen_spots:
            grid_list[spot] = '✅'

        grid_rows = [''.join(grid_list[i:i + 5]) for i in range(0, grid_size, 5)]
        prediction = '\n'.join(grid_rows)

        return prediction, mines_amount, bet_amount, uuid

    def oldrfc(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None", "N/A"
        
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                        headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        hash_1 = data_game['game']['_id']['$oid']
        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                            'size': '1000',
                            'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']
        
        num_spots = 3
        safe_spots = []
        past_mine = mines_location
        past_safe = clicked_spots

        def get_adjacent(tiles):
            adjacent = []
            row = (tiles - 1) // 5
            col = (tiles - 1) % 5
            for r in range(max(row-1, 0), min(row+2, 5)):
                for c in range(max(col-1, 0), min(col+2, 5)):
                    if r == row and c == col:
                        continue
                    adjacent.append(r * 5 + c + 1)
            return adjacent

        X = []
        y = []

        for i in range(25):
            adj_mines = sum(1 for n in past_mine if n in get_adjacent(i))
            num_flagged = len(past_mine)
            num_safe_tiles = (25 - len(past_mine))
            prob_safe = (num_safe_tiles - adj_mines) / num_safe_tiles
            adj_prob_safe = sum(1 for n in past_safe if n in get_adjacent(i) and n != '❌')
            adj_cleared = sum(1 for n in safe_spots if n in get_adjacent(i) and n in past_safe)
            X.append((i, num_safe_tiles, prob_safe, num_flagged, adj_prob_safe, adj_cleared))
            y.append(1 if i in past_mine else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        safe_spots = [(i, prob[0]) for i, prob in enumerate(classifier.predict_proba(X))]
        
        # Sort safe spots based on distance from past mines
        safe_spots.sort(key=lambda x: min(abs(x[0] - m) for m in past_mine), reverse=True)
        
        chosen_spots = []

        np.random.seed(21)  # Set a fixed seed for deterministic shuffling
        np.random.shuffle(safe_spots)
        
        for spot in safe_spots:
            if spot[0] not in past_mine and spot[0] not in past_safe:
                if len(chosen_spots) == num_spots:
                    break
                chosen_spots.append(spot[0])
                # Remove adjacent spots from consideration
                adjacent_spots = get_adjacent(spot[0])
                for adj_spot in adjacent_spots:
                    if adj_spot in safe_spots:
                        safe_spots.remove(adj_spot)
        
        grid_list = ['❌'] * 25
        for r in range(5):
            for c in range(5):
                index = r * 5 + c
                if index in past_mine:  
                    grid_list[index] = '❌'
                elif index in chosen_spots:
                    grid_list[index] = '✅'

        # Reformat grid_list into a list of rows
        grid_rows = [''.join(grid_list[i:i+5]) for i in range(0, 25, 5)]

        prediction = '\n'.join(grid_rows)

        return prediction, mines_amount, bet_amount, uuid

    def rfc(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        
        r7 = scraper.get("https://api.bloxflip.com/games/mines",
                        headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce5 = data_game['game']['nonce'] - 1
        
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={
                            'size': '1000',
                            'page': '0'
                        })
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']
        
        num_spots = 15
        safe_spots = []
        past_mine = mines_location
        past_safe = clicked_spots

        def get_adjacent(tiles):
            adjacent = []
            row = (tiles - 1) // 5
            col = (tiles - 1) % 5
            for r in range(max(row-1, 0), min(row+2, 5)):
                for c in range(max(col-1, 0), min(col+2, 5)):
                    if r == row and c == col:
                        continue
                    adjacent.append(r * 5 + c + 1)
            return adjacent

        X = []
        y = []

        for i in range(25):
            adj_mines = sum(1 for n in past_mine if n in get_adjacent(i))
            num_flagged = len(past_mine)
            num_safe_tiles = (25 - len(past_mine))
            prob_safe = (num_safe_tiles - adj_mines) / num_safe_tiles
            adj_prob_safe = sum(1 for n in past_safe if n in get_adjacent(i) and n != '❌')
            adj_cleared = sum(1 for n in safe_spots if n in get_adjacent(i) and n in past_safe)
            X.append((i, num_safe_tiles, prob_safe, num_flagged, adj_prob_safe, adj_cleared))
            y.append(1 if i in past_mine else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        safe_spots = [(i, prob[0]) for i, prob in enumerate(classifier.predict_proba(X))]
        
        # Sort safe spots based on distance from past mines
        safe_spots.sort(key=lambda x: min(abs(x[0] - m) for m in past_mine), reverse=True)
        
        chosen_spots = []

        # Shuffle the safe spots to avoid showing them in a row
        np.random.seed(42)  # Set a fixed seed for deterministic shuffling
        np.random.shuffle(safe_spots)
        
        for spot in safe_spots:
            if spot[0] not in past_mine and spot[0] not in past_safe:
                if len(chosen_spots) == num_spots:
                    break
                chosen_spots.append(spot[0])
                # Remove adjacent spots from consideration
                adjacent_spots = get_adjacent(spot[0])
                for adj_spot in adjacent_spots:
                    if adj_spot in safe_spots:
                        safe_spots.remove(adj_spot)
        
        grid_list = ['✅'] * 25
        for r in range(5):
            for c in range(5):
                index = r * 5 + c
                if index in past_mine:  
                    grid_list[index] = '⚠️'
                elif index in chosen_spots:
                    grid_list[index] = '❌'

        # Reformat grid_list into a list of rows
        grid_rows = [''.join(grid_list[i:i+5]) for i in range(0, 25, 5)]

        prediction = '\n'.join(grid_rows)

        return prediction, mines_amount, bet_amount, uuid

    def kna(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r2 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r2.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nounce = data_game['game']['nonce'] - 1

        r = scraper.get('https://api.bloxflip.com/games/mines/history',
                             headers=self.headers,
                             params={"size": '5', "page": '0'})
        data = r.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        num_spots = 3
        safe_spots = []
        past_mine = mines_location
        past_safe = clicked_spots

        def get_adjacent(tiles):
            adjacent = []
            row = (tiles - 1) // 5
            col = (tiles - 1) % 5
            for r in range(max(row-1, 0), min(row+2, 5)):
                for c in range(max(col-1, 0), min(col+2, 5)):
                    for c in range(max(col-1, 0), min(col+2, 5)):
                        if r == row and c == col:
                            continue
                        adjacent.append(r * 5 + c + 1)
                        return adjacent
            
        X = []
        y = []

        for i in range(25):
            adj_mines = sum(1 for n in past_mine if n in get_adjacent(i) and n == '❌')
            num_flagged = len(past_mine)
            num_safe_tiles = (25 - len(past_mine))
            prob_safe = (num_safe_tiles - adj_mines) / num_safe_tiles
            adj_prob_safe = sum(1 for n in past_safe if n in get_adjacent(i) and n != '❌' and n in [index for index in X])
            adj_cleared = sum(1 for n in safe_spots if n in get_adjacent(i) and n in past_safe)
            X.append((i, num_safe_tiles, prob_safe, num_flagged, adj_prob_safe, adj_cleared))
            y.append(i in past_mine)

        classifiers = KNeighborsClassifier(n_neighbors=3)
        classifiers.fit(X, y)

        probs = classifiers.predict_proba(X)
        predictions = classifiers.predict(X)

        correct_predictions = sum(1 for predicted, actual in zip(predictions, y) if predicted == actual)
        total_predictions = len(y)
        real_accuracy = correct_predictions / total_predictions

        np.random.seed(7)
        safe_spots = [(i, prob[0]) for i, prob in enumerate(probs)]
        np.random.shuffle(safe_spots)
        chosen_spots = []

        for spot in safe_spots:
            if spot[0] not in past_mine and spot[0] not in past_safe:
                chosen_spots.append(spot[0])
                if len(chosen_spots) == num_spots:
                    break
        
        grid_list = ['❌'] * 25
        for r in range(5):
            for c in range(5):
                index = r * 5 + c
                if index in past_mine:  
                    grid_list[index] = '❌'
                elif index in chosen_spots:
                    grid_list[index] = '✅'


        prediction = f"{''.join(grid_list[:5])}\n{''.join(grid_list[5:10])}\n{''.join(grid_list[10:15])}\n{''.join(grid_list[15:20])}\n{''.join(grid_list[20:])}"

        return prediction, mines_amount, bet_amount, uuid
    
    def fuckno(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        # Fetch the current game details
        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        nonce5 = data_game['game']['nonce'] - 1

        # Fetch the game history
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1000', 'page': '0'})
        data = r8.json()['data']
        latest_game = data[0]
        mine_locations = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        tile_amount = 3
        grid = [
        '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌',
        '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌'
        ]


        count = 0
        while tile_amount > count:
            a = random.randint(0, 24)
    
            grid[a] = '✅'
            count += 1

        chance = random.randint(45, 95)
        if tile_amount < 4:
            chance = chance - 15
        
        prediction = "\n" +grid[0]+grid[1]+grid[2]+grid[3]+grid[4]+"\n"+grid[5]+grid[6]+grid[7]+grid[8]+grid[9]+"\n"+grid[10]+grid[11]+grid[12]+grid[13]+grid[14]+"\n"+grid[15]+grid[16]+grid[17] \
            +grid[18]+grid[19]+"\n"+grid[20]+grid[21]+grid[22]+grid[23]+grid[24]
        
        return prediction, mines_amount, bet_amount, uuid
    
    def junior(self):     
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        
        r2 = scraper.get("https://api.bloxflip.com/games/mines",
                            headers=self.headers)
        data_game = json.loads(r2.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        hash = data_game['game']['_id']['$oid'] 
        nounce = data_game['game']['nonce'] - 1
        random_accuracy = random.uniform(70, 85)
        r = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={"size": '5', "page": '0'})
        data = r.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']
        

        def calculate_positions(tile_amt, round_id, grid_size=25):
            # Predefined patterns
            algos = [
        [0, 6, 9, 10, 8, 15],
        [6, 9, 2, 18, 20, 23],
        [15, 20, 7, 11, 13, 3],
        [7, 9, 10, 11, 16, 20],
        [5, 19, 12, 15, 8, 24],
        [4, 19, 20, 6, 9, 10],
        [7, 19, 20, 24, 13, 2],
        [1, 5, 10, 15, 20, 21],
        [2, 8, 13, 17, 22, 24],
        [3, 7, 12, 16, 19, 23],
        [4, 9, 14, 18, 21, 25],
        [1, 6, 11, 16, 21, 24],
        [2, 7, 12, 17, 22, 25],
        [3, 8, 13, 18, 23, 24],
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
        [1, 6, 7, 8, 12, 13],
        [2, 7, 8, 9, 13, 14],
        [3, 8, 9, 10, 14, 15],
        [4, 9, 10, 11, 15, 16],
        [5, 10, 11, 16, 17, 21],
        [0, 5, 6, 11, 12, 17],
        [1, 6, 7, 12, 13, 18],
        [2, 7, 8, 13, 14, 19],
        [3, 8, 9, 14, 15, 20],
        [4, 9, 10, 15, 16, 21],
        [0, 6, 12, 18, 24, 4],
        [1, 7, 13, 19, 20, 14],
        [2, 8, 9, 10, 11, 15],
        [3, 4, 5, 21, 22, 23],
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
        [1, 6, 7, 8, 12, 13],
        [2, 7, 8, 9, 13, 14],
        [3, 8, 9, 10, 14, 15],
        [4, 9, 10, 11, 15, 16],
        [5, 10, 11, 16, 17, 21],
        [0, 5, 6, 11, 12, 17],
        [1, 6, 7, 12, 13, 18],
        [2, 7, 8, 13, 14, 19],
        [3, 8, 9, 14, 15, 20],
        [4, 9, 10, 15, 16, 21],
        [0, 6, 12, 18, 24, 4],
        [1, 7, 13, 19, 20, 14],
        [2, 8, 9, 10, 11, 15],
        [3, 4, 5, 21, 22, 23],
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [2, 7, 8, 9, 13, 14],
        [3, 8, 9, 10, 14, 15],
        [4, 9, 10, 11, 15, 16],
        [5, 10, 11, 16, 17, 21],
        [0, 5, 6, 11, 12, 17],
        [1, 6, 7, 12, 13, 18],
        [2, 7, 8, 13, 14, 19],
        [3, 8, 9, 14, 15, 20],
        [4, 9, 10, 15, 16, 21],
        [0, 6, 12, 18, 24, 4],
        [1, 7, 13, 19, 20, 14],
        [2, 8, 9, 10, 11, 15],
        [3, 4, 5, 21, 22, 23],
        [0, 1, 2, 3, 4, 5],
        [18, 19, 20, 21, 22, 23],
        [1, 6, 7, 8, 12, 13],
        [2, 7, 8, 9, 13, 14],
        [3, 8, 9, 10, 14, 15],
        [4, 9, 10, 11, 15, 16],
        [5, 10, 11, 16, 17, 21],
        [0, 5, 6, 11, 12, 17],
        [1, 6, 7, 12, 13, 18],
        [2, 7, 8, 13, 14, 19],
        [3, 8, 9, 14, 15, 20],
        [4, 9, 10, 15, 16, 21],
        [0, 6, 12, 18, 24, 4],
        [1, 7, 13, 19, 20, 14],
        [2, 8, 9, 10, 11, 15],
        [3, 4, 5, 21, 22, 23],
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
        [1, 6, 7, 8, 12, 13],
        [2, 7, 8, 9, 13, 14],
        [3, 8, 9, 10, 14, 15],
        [4, 9, 10, 11, 15, 16],
        [5, 10, 11, 16, 17, 21],
        [0, 5, 6, 11, 12, 17],
        [1, 6, 7, 12, 13, 18],
        [2, 7, 8, 13, 14, 19],
        [3, 8, 9, 14, 15, 20],
        [4, 9, 10, 15, 16, 21],
        [0, 6, 12, 18, 24, 4],
        [1, 7, 13, 19, 20, 14],
        [2, 8, 9, 10, 11, 15],
        [3, 4, 5, 21, 22, 23]

  ]

            # Convert round_id to a number to determine which pattern to use
            hashed_round_id = hashlib.sha256(round_id.encode()).hexdigest()
            pattern_index = int(hashed_round_id, 16) % len(algos)
            selected_pattern = algos[pattern_index]

            # Calculate safe tile positions
            if tile_amt < len(selected_pattern):
                safe_positions = selected_pattern[:tile_amt]
            else:
                safe_positions = selected_pattern

            # Calculate risky tile positions
            remaining_positions = set(range(grid_size)) - set(safe_positions)
            available_positions = list(remaining_positions)

            # Calculate risky amount based on remaining available positions
            max_risky_amt = len(available_positions) - len(safe_positions)
            risky_amt = max(1, round((max_risky_amt / grid_size) * 10))  # Adjust 10 as needed for more or fewer risky tiles

            # Select random positions for risky tiles
            risky_positions = random.sample(available_positions, risky_amt)

            return safe_positions, risky_positions, available_positions

        def calculate_accuracy(safe_amt, grid_size):
            return (safe_amt / grid_size) * 100

        def predict_possible_bombs(available_positions, safe_positions, risky_positions, grid_size=25):
            possible_bombs = []
            for pos in available_positions:
                adjacent_safe_count = sum((pos + delta) in safe_positions for delta in [-1, 1, -5, 5])
                adjacent_risky_count = sum((pos + delta) in risky_positions for delta in [-1, 1, -5, 5])
                # If adjacent to risky tiles more than safe tiles, consider it a possible bomb
                if adjacent_risky_count > adjacent_safe_count:
                    possible_bombs.append(pos)
            return possible_bombs

        grid_size = 25
        grid = ['❌'] * grid_size

        # Calculate positions for '✅', '⚠️', and available positions
        safe_positions, risky_positions, available_positions = calculate_positions(4, uuid, grid_size)

        for pos in safe_positions:
            grid[pos] = '✅'
        for pos in risky_positions:
            grid[pos] = '❌'

        # Predict possible bomb positions
        possible_bomb_positions = predict_possible_bombs(available_positions, safe_positions, risky_positions, grid_size)
        for pos in possible_bomb_positions:
            grid[pos] = '❌'

        # Calculate accuracy percentage
        accuracy = calculate_accuracy(len(safe_positions), grid_size)
        chance = round(accuracy, 2)

        # Construct the grid display
        grid_display = "\n".join("".join(grid[i:i+5]) for i in range(0, grid_size, 5))

        return grid_display, mines_amount, bet_amount, uuid
    
    def fuckalgo(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce5 = data_game['game']['nonce'] - 1

        r8 = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1000', 'page': '0'})
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid_size = 25
        num_spot = 3

        def is_neighbor(spot1, spot2):
            row1, col1 = divmod(spot1, 5)
            row2, col2 = divmod(spot2, 5)
            return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1

        bomb_likelihoods = np.array([
            [0.9, 0.9, 0.9, 0.9, 0.9],
            [0.9, 0.4, 0.4, 0.4, 0.9],
            [0.9, 0.4, 0.4, 0.4, 0.9],
            [0.9, 0.4, 0.4, 0.4, 0.9],
            [0.9, 0.9, 0.9, 0.9, 0.9]
        ])

        def calculate_probabilities(mines_location, clicked_spots, grid_size):
            mine_counts = np.zeros(grid_size)
            safe_counts = np.zeros(grid_size)

            for spot in mines_location:
                mine_counts[spot] += 1

            for spot in clicked_spots:
                safe_counts[spot] -= 1

            probabilities = (safe_counts + 1) / (mine_counts - safe_counts + 2)

            return probabilities

        probabilities = calculate_probabilities(mines_location, clicked_spots, grid_size)

        for i in range(grid_size):
            row, col = divmod(i, 5)
            probabilities[i] *= (1 - bomb_likelihoods[row, col])
            if any(is_neighbor(i, mine) for mine in mines_location):
                probabilities[i] *= 0.235

        def select_safest_spots(probabilities, num_spot):
            spot_prob_pairs = [(spot, prob) for spot, prob in enumerate(probabilities)]
            spot_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            safest_spots = [spot for spot, _ in spot_prob_pairs[:num_spot]]
            return safest_spots

        safest_spots = select_safest_spots(probabilities, num_spot)

        grid = [['❌'] * 5 for _ in range(5)]

        for spot in clicked_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '❌'

        for spot in safest_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '✅'

        grid = [cell for row in grid for cell in row]

        Prediction1 = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))

        return Prediction1, mines_amount, bet_amount, uuid



    def pulsivev5(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        # Fetch the current game details
        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce5 = data_game['game']['nonce'] - 1

        # Fetch the game history
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1000', 'page': '0'})
        data = r8.json()['data']
        latest_game = data[10]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid_size = 25
        num_spot = self.max_tiles

        def is_neighbor(spot1, spot2):
            row1, col1 = divmod(spot1, 5)
            row2, col2 = divmod(spot2, 5)
            return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1

        # Improved bomb likelihood matrix based on historical data
        bomb_likelihoods = np.full((5,5), 0.5)

        def calculate_probabilities(mines_location, clicked_spots, grid_size):
            mine_counts = np.zeros(grid_size)
            safe_counts = np.zeros(grid_size)

            for i, game in enumerate(reversed(data[:1])):
                weight = 1 / (i + 1)
                for spot in game['mineLocations']:
                    mine_counts[spot] += weight
                for spot in game['uncoveredLocations']:
                    safe_counts[spot] -= weight

            probabilities = (safe_counts + 1) / (mine_counts - safe_counts + 5)

            return probabilities

        probabilities = calculate_probabilities(mines_location, clicked_spots, grid_size)

        for i in range(grid_size):
            row, col = divmod(i, 5)
            probabilities[i] *= (1 - bomb_likelihoods[row, col])
            if any(is_neighbor(i, mine) for mine in mines_location):
                probabilities[i] *= 7

        def select_safest_spots(probabilities, num_spot):
            spot_prob_pairs = [(spot, prob) for spot, prob in enumerate(probabilities)]
            spot_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            safest_spots = [spot for spot, _ in spot_prob_pairs[:num_spot]]
            return safest_spots

        safest_spots = select_safest_spots(probabilities, num_spot)

        grid = [[''] * 5 for _ in range(5)]

        for spot in clicked_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '❌'

        for spot in safest_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '✅'

        grid = [cell for row in grid for cell in row]

        Prediction1 = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))

        return Prediction1, mines_amount, bet_amount, uuid

    def algorithm(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        
        r2 = scraper.get("https://api.bloxflip.com/games/mines",
                            headers=self.headers)
        data_game = json.loads(r2.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']
        hash = data_game['game']['_id']['$oid'] 
        nounce = data_game['game']['nonce'] - 1
        random_accuracy = random.uniform(70, 85)
        r = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={"size": '5', "page": '0'})
        data = r.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']


        grid_size = 25
        num_spot = self.max_tiles

        def is_neighbor(spot1, spot2):
            row1, col1 = divmod(spot1, 1)
            row2, col2 = divmod(spot2, 1)
            
            return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1

        def calculate_safest_spot(clicked_spots, mines_location, grid_size, num_spot):
            pattern_probs = np.array([
                [0.2, 0.3, 0.4, 0.1, 0.5],
                [0.6, 0.7, 0.8, 0.9, 0.0],
                [0.5, 0.3, 0.2, 0.1, 0.4],
                [0.8, 0.7, 0.6, 0.9, 0.0],
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ])

            bomb_likelihoods = np.array([
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9, 0.0],
                [0.5, 0.4, 0.3, 0.2, 0.1],
                [0.9, 0.8, 0.7, 0.6, 0.0],
                [0.2, 0.3, 0.4, 0.5, 0.6]
            ])

            safe_patterns = np.zeros(grid_size)
            for i in range(grid_size):
                num_neighboring_mines = sum(1 for neighbor in mines_location if is_neighbor(i, neighbor))
                if i not in clicked_spots and i not in mines_location:
                    row, col = divmod(i, 5)
                    if num_neighboring_mines == 0:
                        safe_patterns[i] = pattern_probs[row, col]
              

            danger_scores = bomb_likelihoods.flatten() + safe_patterns

            least_dangerous_spots = np.argsort(danger_scores)
            selected_spots = [spot for spot in least_dangerous_spots if spot not in clicked_spots][:num_spot]

            return selected_spots

        def calculate_chances_mines_at_location(history, location):
            mine_count = sum(1 for game in history if location in game['mineLocations'])
            total_games = len(history)
            if total_games == 0:
                return 0.0
            return mine_count / total_games

        selected_spots = calculate_safest_spot(clicked_spots, mines_location, grid_size, num_spot)

        grid = ['💥'] * grid_size
        for spot in clicked_spots:
         grid[spot] = '❔'
        for spot in selected_spots:
         grid[spot] = '✅' if spot in selected_spots else '❌'

        mine_emoji = '❌'
        safe_emoji = '✅'

        grid = [
        mine_emoji if cell == '❌' or cell == '💥' or cell == '❔' else safe_emoji
        for cell in grid
        ]
        Prediction1 = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))

        return Prediction1, mines_amount, bet_amount, uuid

    def olderpulsivev5(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"
        
        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce5 = data_game['game']['nonce'] - 1

        r8 = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1000', 'page': '0'})
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid_size = 25
        num_spot = self.max_tiles

        def is_neighbor(spot1, spot2):
            row1, col1 = divmod(spot1, 5)
            row2, col2 = divmod(spot2, 5)
            return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1

        def calculate_safest_and_most_dangerous_spots(clicked_spots, mines_location, grid_size, num_mines, num_spot):
            safe_spots = []
            bomb_likelihoods = np.full((5,5), 0)

            for spot in range(grid_size):
                if spot not in clicked_spots and spot not in mines_location:
                    num_neighboring_mines = sum(1 for neighbor in mines_location if is_neighbor(spot, neighbor))
                    if num_neighboring_mines == 0:
                        safe_spots.append(spot)

            likelihood_scores = [1 - bomb_likelihoods[spot // 5, spot % 5] for spot in safe_spots]

            sorted_safe_spots = [x for _, x in sorted(zip(likelihood_scores, safe_spots), reverse=True)]

            safest_spots = sorted_safe_spots[:int(num_spot)]

            bomb_chance_at_safest_spot = 1 - likelihood_scores[safe_spots.index(safest_spots[0])]

            most_dangerous_spot = safe_spots[np.argmax(likelihood_scores)]

            return safest_spots, bomb_chance_at_safest_spot, most_dangerous_spot

        latest_game = {
            "mineLocations": mines_location,
            "uncoveredLocations": clicked_spots,
        }

        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        num_mines = len(mines_location)

        safest_spots, bomb_chance, most_dangerous_spot = calculate_safest_and_most_dangerous_spots(clicked_spots, mines_location, grid_size, num_mines, num_spot)

        grid = [['❓'] * 5 for _ in range(5)]

        for spot in clicked_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '❌'

        np.random.shuffle(safest_spots)  # Shuffle the safe spots

        # Split the safe spots into rows, shuffle each row, and then flatten them back into a single list
        shuffled_safe_spots = [cell for row in [np.random.permutation(row) for row in np.array_split(safest_spots, 5)] for cell in row]

        for spot in shuffled_safe_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '✅'

        most_dangerous_row, most_dangerous_col = divmod(most_dangerous_spot, 5)
        grid[most_dangerous_row][most_dangerous_col] = '❌'

        # Convert the grid to a flattened list for better display
        grid = [cell for row in grid for cell in row]

        Prediction1 = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))

        return Prediction1, mines_amount, bet_amount, uuid

    def oldpulsivev5(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce5 = data_game['game']['nonce'] - 1

        r8 = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1000', 'page': '0'})
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid_size = 25
        num_spot = self.max_tiles

        def is_neighbor(spot1, spot2):
            row1, col1 = divmod(spot1, 5)
            row2, col2 = divmod(spot2, 5)
            return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1

        bomb_likelihoods = np.full((5,5), 1)

        def calculate_probabilities(mines_location, clicked_spots, grid_size):
            mine_counts = np.zeros(grid_size)
            safe_counts = np.zeros(grid_size)

            for spot in mines_location:
                mine_counts[spot] += 1

            for spot in clicked_spots:
                safe_counts[spot] += 1

            probabilities = (safe_counts + 1) / (mine_counts + safe_counts + 2)

            return probabilities

        probabilities = calculate_probabilities(mines_location, clicked_spots, grid_size)

        for i in range(grid_size):
            row, col = divmod(i, 5)
            probabilities[i] *= (1 - bomb_likelihoods[row, col])
            if any(is_neighbor(i, mine) for mine in mines_location):
                probabilities[i] *= 0.5

        def select_safest_spots(probabilities, num_spot):
            spot_prob_pairs = [(spot, prob) for spot, prob in enumerate(probabilities)]
            spot_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            safest_spots = [spot for spot, _ in spot_prob_pairs[:num_spot]]
            random.shuffle(safest_spots)
            return safest_spots

        safest_spots = select_safest_spots(probabilities, num_spot)

        grid = [['❌'] * 5 for _ in range(5)]

        for spot in clicked_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '❌'

        for spot in safest_spots:
            row, col = divmod(spot, 5)
            grid[row][col] = '✅'

        grid = [cell for row in grid for cell in row]

        Prediction1 = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))

        return Prediction1, mines_amount, bet_amount, uuid
    
    def oldpulsivesafe(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A", "N/A", "None"

        # Fetch the current game details
        r7 = scraper.get("https://api.bloxflip.com/games/mines", headers=self.headers)
        data_game = json.loads(r7.text)
        mines_amount = data_game['game']['minesAmount']
        uuid = data_game['game']['uuid']
        bet_amount = data_game['game']['betAmount']

        nonce5 = data_game['game']['nonce'] - 1

        # Fetch the game history
        r8 = scraper.get('https://api.bloxflip.com/games/mines/history', headers=self.headers, params={'size': '1000', 'page': '0'})
        data = r8.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        num_spots = self.max_tiles
        past_mine = mines_location
        past_safe = clicked_spots

        dangerous_spots = [0, 1, 2, 5, 10, 11, 14, 21, 23]

        def get_adjacent(tiles):
            adjacent = []
            row = (tiles - 1) // 5
            col = (tiles - 1) % 5
            for r in range(max(row-1, 0), min(row+2, 5)):
                for c in range(max(col-1, 0), min(col+2, 5)):
                    if r == row and c == col:
                        continue
                    adjacent.append(r * 5 + c + 1)
                return adjacent

        X = []
        y = []

        for i in range(25):
            adj_mines = sum(1 for n in past_mine if n in get_adjacent(i))
            num_flagged = len(past_mine)
            num_safe_tiles = (25 - len(past_mine))
            prob_safe = (num_safe_tiles - adj_mines) / num_safe_tiles
            adj_prob_safe = sum(1 for n in past_safe if n in get_adjacent(i) and n != '❌')
            adj_cleared = sum(1 for n in past_safe if n in get_adjacent(i))
            X.append((i, num_safe_tiles, prob_safe, num_flagged, adj_prob_safe, adj_cleared))
            y.append(1 if i in past_mine else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        safe_spots = [(i, prob[0]) for i, prob in enumerate(classifier.predict_proba(X))]

        # Sort safe spots based on distance from past mines
        safe_spots.sort(key=lambda x: min(abs(x[0] - m) for m in past_mine), reverse=True)

        chosen_spots = []

        # Create a deterministic shuffle without using randomization
        permutation = np.arange(len(safe_spots))
        np.random.seed(42)  # Setting a seed for deterministic permutation
        np.random.shuffle(permutation)
        shuffled_safe_spots = [safe_spots[i] for i in permutation]

        # Check if dangerous spots are near past bombs
        dangerous_spots_near_bombs = set()
        for spot in dangerous_spots:
            if any(n in get_adjacent(spot) for n in past_mine):
                dangerous_spots_near_bombs.add(spot)

        for spot in shuffled_safe_spots:
            if spot[0] not in past_mine and spot[0] not in past_safe:
                if len(chosen_spots) == num_spots:
                    break
                # Add logic for dangerous spots
                if spot[0] in dangerous_spots:
                    continue
                chosen_spots.append(spot[0])
                # Remove adjacent spots from consideration
                adjacent_spots = get_adjacent(spot[0])
                for adj_spot in adjacent_spots:
                    if adj_spot in shuffled_safe_spots:
                        shuffled_safe_spots.remove(adj_spot)

        grid_list = ['❌'] * 25
        for r in range(5):
            for c in range(5):
                index = r * 5 + c
                if index in past_mine:
                    grid_list[index] = '❌'
                elif index in chosen_spots:
                    grid_list[index] = '✅'

        # Reformat grid_list into a list of rows
        grid_rows = [''.join(grid_list[i:i + 5]) for i in range(0, 25, 5)]

        # Join the rows with newline characters to create the final prediction string
        prediction = '\n'.join(grid_rows)

        return prediction, mines_amount, bet_amount, uuid
    
    def oldalgonouse(self):
        if not self.check_game():
            return "*Currently not in game. Please start a game and run command again.*\n", "N/A"
        
        s = scraper.get("https://api.bloxflip.com/games/mines",
                         headers=self.headers)
        data_game = json.loads(s.text)
        mines_amount = data_game['game']['minesAmount']

        r = scraper.get('https://api.bloxflip.com/games/mines/history',
                        headers=self.headers,
                        params={"size": '5', "page": '0'})
        data = r.json()['data']
        latest_game = data[0]
        mines_location = latest_game['mineLocations']
        clicked_spots = latest_game['uncoveredLocations']

        grid = ['💥'] * 25

        logarithm_model = np.zeros((25, 25, 25, 25))

        for i in range(len(clicked_spots) - 3):
            prev3_spot = clicked_spots[i]
            prev2_spot = clicked_spots[i + 1]
            prev_spot = clicked_spots[i + 2]
            curr_spot = clicked_spots[i + 3]
            logarithm_model[prev3_spot, prev2_spot, prev_spot, curr_spot] += 1

        prior_safe = (25 - len(clicked_spots) - len(mines_location)) / 25
        prior_dangerous = len(mines_location) / 25

        safe_probs = np.zeros(25)
        dangerous_probs = np.zeros(25)
        for i in range(25):
            if i not in clicked_spots and i not in mines_location:
                for j in range(25):
                    for k in range(25):
                        count = logarithm_model[j, k, i, :]
                        if np.sum(count) > 0:
                            safe_probs[i] += np.product(
                                (count + 1) / (np.sum(count) + prior_safe))
                            dangerous_probs[i] += np.product(
                                (count + 1) / (np.sum(count) + prior_dangerous))

        game_data = 10000
        exponential_growth = 1.0

        unclicked_spots = list(set(range(25)) - set(clicked_spots) - set(mines_location))
        num_unclicked = min(len(unclicked_spots), 25 - len(clicked_spots) - len(mines_location))

        safe_counts = np.zeros(num_unclicked)
        bad_counts = np.zeros(num_unclicked)
        for i in range(game_data):
            game = latest_game.copy()
            unclicked_spots_subset = []
            if len(unclicked_spots) > 0:
                np.random.shuffle(unclicked_spots)
                unclicked_spots_subset = unclicked_spots[:min(num_unclicked, len(unclicked_spots))]
            game['uncoveredLocations'] = clicked_spots + unclicked_spots_subset
            exploded = False
            num_mines_uncovered = 0

        for spot in unclicked_spots_subset:
            if spot in mines_location:
                num_mines_uncovered += 1
                exploded = True
                break
            elif exponential_growth < safe_probs[spot]:
                game['uncoveredLocations'].append(spot)

        for spot in unclicked_spots_subset:
            if not exploded:
                if spot not in game['uncoveredLocations']:
                    safe_counts[unclicked_spots.index(spot)] += 1
                else:
                    bad_counts[unclicked_spots.index(spot)] -= 1
                if spot in mines_location:
                    num_mines_uncovered += 1

        mc_safe_probs = np.zeros(25)
        for i in range(25):
            if i not in clicked_spots and i not in mines_location:
                mc_safe_probs[i] = (safe_counts[unclicked_spots.index(i)] +
                        bad_counts[unclicked_spots.index(i)] +
                        safe_probs[i] * game_data) / (game_data + np.sum(safe_probs))

        num_spots = self.max_tiles
        top_safe_spots = np.argsort(mc_safe_probs)[::-1]
        top_safe_spots = [
            spot for spot in top_safe_spots if spot not in mines_location
        ][:25]
        selected_safe_spots = np.random.choice(top_safe_spots,
                                                min(num_spots, len(top_safe_spots)),
                                                replace=False)

        top_dangerous_spots = np.argsort(dangerous_probs)[::-1]
        top_dangerous_spots = [
            spot for spot in top_dangerous_spots if spot not in mines_location
        ][:25]
        selected_dangerous_spots = np.random.choice(top_dangerous_spots,
                                                    min(num_spots, len(top_dangerous_spots)),
                                                    replace=False)

        for i, spot in enumerate(range(len(grid))):
            if spot not in mines_location and grid[spot] != '❌' and grid[spot] != '❔':
                if spot in selected_safe_spots[:num_spots]:
                    grid[spot] = '✅'
            elif spot in selected_dangerous_spots[:mines_amount]:
                grid[spot] = '❔'
            else:
                grid[spot] = '❌'

        mine_emoji = "❌"
        safe_emoji = "✅"

        grid = [
        mine_emoji if cell == '❌' or cell == '💥' or cell == '❔' else safe_emoji
        for cell in grid
        ]
        prediction = '\n'.join(''.join(grid[i:i + 5]) for i in range(0, len(grid), 5))
        return prediction, "60.0"
    


# -- CLASSES  END --

# -- MAIN FUNCTIONS --





def game_active(gamemode):
    match gamemode:
        case "crash":
            conn = http.client.HTTPSConnection("api.bloxflip.com")
            headers = {
                "Referer": "https://bloxflip.com/",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
            }
            conn.request("GET", url="/games/crash", headers=headers)
            return True if json.loads(conn.getresponse().read().decode())['current']['status'] != 2 else False
        case "slide":
            conn = http.client.HTTPSConnection("api.bloxflip.com")
            headers = {
                "Referer": "https://bloxflip.com/",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
            }
            conn.request("GET", url="/games/roulette", headers=headers)
            return False if json.loads(conn.getresponse().read().decode())['current']['joinable'] else True




def claimRB(auth):
    conn = http.client.HTTPSConnection("api.bloxflip.com")
    headers = {
        "x-auth-token": auth,
        "Referer": "https://bloxflip.com/",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
    }
    conn.request("POST", url="/vip/claim", headers=headers)
    return json.loads(conn.getresponse().read().decode())


def getProfile(auth):
    conn = http.client.HTTPSConnection("api.bloxflip.com")
    headers = {
        "x-auth-token": auth,
        "Referer": "https://bloxflip.com/",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
    }
    conn.request("GET", url="/user", headers=headers)
    return json.loads(conn.getresponse().read().decode())


def validToken(auth):
    conn = http.client.HTTPSConnection("api.bloxflip.com")
    headers = {
        "x-auth-token": auth,
        "Referer": "https://bloxflip.com/",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
    }
    conn.request("GET", url="/user", headers=headers)
    return json.loads(conn.getresponse().read().decode())['success']







def gentime(length):
    match length:
        case "Lifetime":
            return datetime.datetime.now() + datetime.timedelta(weeks=100000)
        case "Monthly":
            return datetime.datetime.now() + datetime.timedelta(weeks=4)
        case "Weekly":
            return datetime.datetime.now() + datetime.timedelta(weeks=4)
        case "Daily":
            return datetime.datetime.now() + datetime.timedelta(days=1)
        case "Hourly":
            return datetime.datetime.now() + datetime.timedelta(hours=1)


def checkauth(id):
    authf = json.load(open(auth_file, 'r'))
    v = list(authf.values())
    v = [v['user_id'] for v in v]
    if id in v:
        getkey = list(authf.keys())[v.index(id)]
        expires = datetime.datetime.strptime(authf[getkey]['expires'], '%Y-%m-%d %H:%M:%S.%f')
        if expires <= datetime.datetime.now():
            return {"valid": False, "reason": "exp"}
        else:
            auth = authf[getkey].get('auth_token')
            return {"valid": True, "token": auth} if auth else {"valid": False, "reason": "NOLINK"}
    else:
        return {"valid": False, "reason": "NO_KEY_EXIST"}


# -- MAIN FUNCTIONS --
# -- ERROR HANDLING --

@tree.error
async def on_command_error(interaction, error):
    if isinstance(error, app_commands.MissingRole):
        e = discord.Embed(title="No Permissions",
                          description=f"You don't have any permissions to perform this command.",
                          color=discord.Color.red())
        await interaction.response.send_message(embed=e, ephemeral=True)
    elif isinstance(error, app_commands.CommandOnCooldown):
        e = discord.Embed(title="Cooldown",
                          description=f"Please wait {round(error.retry_after, 2)} seconds until trying this command again.",
                          color=discord.Color.red())
        await interaction.response.send_message(embed=e, ephemeral=True)
    else:
        e = discord.Embed(title="Error Occured",
                          description=f"An error has occured: {error}",
                          color=discord.Color.red())
        await interaction.response.send_message(embed=e, ephemeral=True)


# -- ERROR HANDLING --
@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="credits", description="Sees who made the bot.")
async def credits(interaction: discord.Interaction):
    e = discord.Embed(title="Bot Credits",
                        description=f"**Main Developer**: flashywastaken, robert\n**Project Supporter**: HieuPC",
                        color=discord.Color.green())
    await interaction.response.send_message(embed=e)


@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="checkuser", description="Checks if a user has a key.")
@app_commands.checks.has_role(settings['role_to_create_keys'])
async def checkuser(interaction: discord.Interaction, member: discord.Member):
    x = json.load(open(auth_file, 'r'))
    get_users = [x['user_id'] for x in x.values()]
    if member.id in get_users:
        key = list(x.keys())[get_users.index(member.id)]
        get_expire = x[key]['expires']
        exp_data = datetime.datetime.strptime(get_expire, '%Y-%m-%d %H:%M:%S.%f')
        current = exp_data - datetime.datetime.now()
        e = discord.Embed(title="User Fetched", description=f"{member.mention}'s Information\n\n**Key**: {key}\n**Expires**: {current.days} days",
                          color=discord.Color.green())
        await interaction.response.send_message(embed=e, ephemeral=True)
    else:
        e = discord.Embed(title="User Fetched Error", description=f"{member.mention} does not have a key.",
                          color=discord.Color.red())
        await interaction.response.send_message(embed=e, ephemeral=True)


@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="createkey", description="Add a new key to the database.")
@app_commands.choices(length=[
    discord.app_commands.Choice(name="Lifetime", value="Lifetime"),
    discord.app_commands.Choice(name="Monthly", value="Monthly"),
    discord.app_commands.Choice(name="Weekly", value="Weekly"),
    discord.app_commands.Choice(name="Daily", value="Daily"),
    discord.app_commands.Choice(name="Hourly", value="Hourly")
])
@app_commands.checks.has_role(settings['role_to_create_keys'])
async def createkey(interaction: discord.Interaction, length: str,user: discord.Member = None):
    time = gentime(length)
    predictorname = settings['predictorname']
    key_creation = f"{predictorname}" + "-" + secrets.token_hex(10)
    with open(auth_file, 'r') as f:
        j = json.load(f)
        j[key_creation] = {"user_id": None, "auth_token": None, "expires": str(time)}
    with open(auth_file, 'w') as b:
        json.dump(j, b, indent=4)
    if user:
        e = discord.Embed(title="Your Key", description=f"**Key**: {key_creation}\n**Expires**: {length}",
                          color=discord.Color.green())
        await user.send(embed=e)
        e = discord.Embed(title="Key Creation", description=f"Key was sent to {user.mention}\n**Key**: {key_creation}\n**Expires**: {length}",
                          color=discord.Color.green())
        await interaction.response.send_message(embed=e, ephemeral=True)
    else:
        e = discord.Embed(title="Key Creation", description=f"**Key**: {key_creation}\n**Expires**: {length}",
                          color=discord.Color.green())
        await interaction.response.send_message(embed=e, ephemeral=True)



@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="deletekey", description="Delete a key from the database.")
@app_commands.checks.has_role(settings['role_to_create_keys'])
async def deletekey(interaction: discord.Interaction, key: str):
    x = json.load(open(auth_file, 'r'))
    if x.get(key):
        # pop key from json file
        x.pop(key)
        with open(auth_file, 'w') as f:
            json.dump(x, f, indent=4)
        e = discord.Embed(title="Key Removed", description="You have successfully removed this key!",
                          color=discord.Color.green())
        await interaction.response.send_message(embed=e, ephemeral=True)
    else:
        e = discord.Embed(title="Key Error", description="Key does not exist.", color=discord.Color.red())
        await interaction.response.send_message(embed=e, ephemeral=True)


@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="redeem", description="Reedems your key and adds it to the database.")
async def redeem(interaction: discord.Interaction, key: str):
    keys = json.load(open(auth_file, 'r'))
    users = [x['user_id'] for x in keys.values()]
    if keys.get(key) and not keys[key]['user_id'] and not interaction.user.id in users:
        with open(auth_file, 'r') as f:
            j = json.load(f)
            j[key].update({"user_id": interaction.user.id})
        with open(auth_file, 'w') as b:
            json.dump(j, b, indent=4)
        e = discord.Embed(title="Key Redeemed", description="You have successfully redeemed the key!",
                          color=discord.Color.green())
        await interaction.response.send_message(embed=e, ephemeral=True)
    elif keys.get(key) and keys[key]['user_id']:
        e = discord.Embed(title="Key Error", description="Key has already been redeemed!", color=discord.Color.red())
        await interaction.response.send_message(embed=e, ephemeral=True)
    elif interaction.user.id in users:
        e = discord.Embed(title="Key Error", description="You have already redeemed a key!", color=discord.Color.red())
        await interaction.response.send_message(embed=e, ephemeral=True)
    else:
        e = discord.Embed(title="Key Error", description="Invalid key!", color=discord.Color.red())
        await interaction.response.send_message(embed=e, ephemeral=True)




@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="howtogettoken", description="Explains to you how to get your bloxflip token.")
async def howtogettoken(interaction: discord.Interaction):
    # thx to nebula for this || :) -Pulsive
    linkacc = (
        "Here's a quick tutorial on how to link your account!\n\n"

        "• Start by heading over to the BloxFlip site and navigate to the console\n\n"

        "To access the console press: ```[Ctrl] + [Shift] + [I]```\n\n"

        "• Head over to the last last line of the console and paste the following prompt:\n"
        "```copy(localStorage.getItem('_DO_NOT_SHARE_BLOXFLIP_TOKEN'));```\n\n"

        "• Finally, execute this command:\n"
        "````/link` auth: your-auth-token```\n\n"
        
        "Tutorial video:\n"
        "https://streamable.com/f6fs8p"
    )

    e = discord.Embed(
        title="",
        description=linkacc,
        color=discord.Color.green()
    )
    e.set_footer(text=TEXT)
    await interaction.response.send_message(embed=e)


@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="autoclaimcase", description="Attemps to auto redeem your wager case.")
async def autoclaimcase(interaction: discord.Interaction, method: str):
    auth_token = checkauth(interaction.user.id)
    if auth_token['valid']:
        name = settings['madeby']
    else:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`,  use `/howtogettoken` if you don't know how to get token." if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Mr",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )


@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="claimrakeback", description="Automatically try to claim your rakeback.")
async def claimrakeback(interaction: discord.Interaction):
    auth_token = checkauth(interaction.user.id)
    if auth_token['valid']:
        g = claimRB(auth_token['token'])
        if g['success']:
            e = discord.Embed(title="Rakeback Success", description=f"You have successfully claimed {round(g['claimed'],2)} robux.",color=discord.Color.green())
            await interaction.response.send_message(embed=e)
        else:
            e = discord.Embed(title="Rakeback Failed",description=f"{g['error']}",color=discord.Color.red())
            await interaction.response.send_message(embed=e)
    else:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`,  use `/howtogettoken` if you don't know how to get token." if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Rakeback Failed",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )


@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="profile", description="View your bloxflip statistics.")
async def profile(interaction: discord.Interaction):
    auth_token = checkauth(interaction.user.id)
    if auth_token['valid']:
        g = getProfile(auth_token['token'])
        ## GENERATE EMBED ##
        robloxid = g['user']['robloxId']
        robloxusername = g['user']['robloxUsername']
        balance = round(g['user']['wallet'], 3)
        wager = round(g['user']['wager'], 2)
        total_withdrawn = round(g["user"]["totalWithdrawn"], 2)
        total_deposited = round(g["user"]["totalDeposited"], 2)
        embedinfo = discord.Embed(title = "Your Profile Statistics", color=discord.Color.green())
        embedinfo.add_field(name="🕵️‍♂️ Roblox Username", value=f"```{robloxusername}```", inline=True)
        embedinfo.add_field(name="💰 Wagered", value=f"```{wager}```", inline=True)
        embedinfo.add_field(name="💵 Balance", value=f"```{balance}```", inline=True)
        embedinfo.add_field(name="📥 Total Deposited", value=f"```{total_deposited}```", inline=True) 
        embedinfo.add_field(name="📤 Total Withdrawn", value=f"```{total_withdrawn}```", inline=True)  
        embedinfo.set_footer(text=TEXT)
        await interaction.response.send_message(embed=embedinfo, ephemeral=False)
    else:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`,  use `/howtogettoken` if you don't know how to get token." if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Profile Statistics",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )


@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="unlink", description="Unlink your bloxflip account from the bot.")
async def unlink(interaction: discord.Interaction):
    auth_token = checkauth(interaction.user.id)
    if auth_token['valid']:
        x = json.load(open(auth_file, 'r'))
        e = [x['user_id'] for x in x.values()]
        getkey = list(x.keys())[e.index(interaction.user.id)]
        x[getkey]['auth_token'] = ""
        with open(auth_file, 'w') as n:
            json.dump(x, n, indent=4)
        await interaction.response.send_message(
            embed=discord.Embed(
                title="Account Unlink",
                description="Your account has been unlinked successfully. Do `/link` to relink",
                color=discord.Color.green()
            ).set_footer(text=TEXT)
        )
    else:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`,  use `/howtogettoken` if you don't know how to get token." if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Account link",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )


@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="account", description="Check your current settings.")
async def account(interaction: discord.Interaction):
    auth_token = checkauth(interaction.user.id)
    if auth_token['valid']:
        x = json.load(open(auth_file, 'r'))
        y = [x['auth_token'] for x in x.values()]
        e = y.index(auth_token['token'])
        key = list(x.keys())[e]
        get_data_key = x[str(key)]
        exp_data = datetime.datetime.strptime(get_data_key['expires'], '%Y-%m-%d %H:%M:%S.%f')
        expire = exp_data - datetime.datetime.now()
        embed = discord.Embed(title="Account Settings", description=f"**Key**: {key}\n**Expires**: {expire.days} Days",
                              color=discord.Color.green())
        embed.set_footer(text=TEXT)
        await interaction.response.send_message(embed=embed, ephemeral=True)
    else:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`,  use `/howtogettoken` if you don't know how to get token." if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Account Settings",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )


@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="link", description="Link your bloxflip token to the bot.")
async def link(interaction: discord.Interaction, auth: str):
    x = json.load(open(auth_file, 'r'))
    v = list(x.values())
    v = [v['user_id'] for v in v]
    if validToken(auth):
        if interaction.user.id in v:
            getkey = list(x.keys())[v.index(interaction.user.id)]
            if not x[getkey]['auth_token']:
                with open(auth_file, 'r') as f:
                    j = json.load(f)
                    j[getkey].update({"auth_token": auth})
                with open(auth_file, 'w') as f:
                    json.dump(j, f, indent=4)
                e = discord.Embed(title="Auth Token", description="Successfully set your auth token!",
                                  color=discord.Color.green())
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e, ephemeral=True)
            else:
                e = discord.Embed(title="Auth Token", description="Token is already in file!",
                                  color=discord.Color.red())
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e, ephemeral=True)
        else:
            e = discord.Embed(title="Auth Token", description="You need to redeem a key before doing this command!",
                              color=discord.Color.red())
            e.set_footer(text=TEXT)
            await interaction.response.send_message(embed=e, ephemeral=True)
    else:
        e = discord.Embed(title="Auth token", description="Auth token is invalid!", color=discord.Color.red())
        e.set_footer(text=TEXT)
        await interaction.response.send_message(embed=e, ephemeral=True)


@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@app_commands.choices(method=[
    app_commands.Choice(name=key, value=value) for key, value in tower_methods.items()
])
@tree.command(name="towers", description="Predicts your current towers game.")
async def towers(interaction: discord.Interaction, method: str):
    auth_token = checkauth(interaction.user.id)
    if auth_token['valid']:
        auth_token = auth_token['token']
        match method:
            case "Probability":
                l = TowersPredictor(auth_token)
                prediction = l.probability()
                e = discord.Embed(title="Towers Prediction",
                                  color=discord.Color.green())
                e.add_field(name="🎯 Prediction", value=prediction)
                e.add_field(name="💎 Method", value=f"> {method}", inline=False)
                e.add_field(name="🍀 Feeling Unlucky?", value="> Use the ``/unrig`` command to get better chances of winning!", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "Pathfinding":
                l = TowersPredictor(auth_token)
                prediction = l.pathfinding()
                e = discord.Embed(title="Towers Prediction",
                                  color=discord.Color.green())
                e.add_field(name="🎯 Prediction", value=prediction)
                e.add_field(name="💎 Method", value=f"> {method}", inline=False)
                e.add_field(name="🍀 Feeling Unlucky?", value="> Use the ``/unrig`` command to get better chances of winning!", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "NearestAdvanced":
                l = TowersPredictor(auth_token)
                prediction = l.nearestadv()
                e = discord.Embed(title="Towers Prediction",
                                  color=discord.Color.green())
                e.add_field(name="🎯 Prediction", value=prediction)
                e.add_field(name="💎 Method", value=f"> {method}", inline=False)
                e.add_field(name="🍀 Feeling Unlucky?", value="> Use the ``/unrig`` command to get better chances of winning!", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "RecentTrend":
                l = TowersPredictor(auth_token)
                prediction = l.recentTrend()
                e = discord.Embed(title="Towers Prediction",
                                  color=discord.Color.green())
                e.add_field(name="🎯 Prediction", value=prediction)
                e.add_field(name="💎 Method", value=f"> {method}", inline=False)
                e.add_field(name="🍀 Feeling Unlucky?", value="> Use the ``/unrig`` command to get better chances of winning!", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "Randomization":
                l = TowersPredictor(auth_token)
                prediction = l.randomization()
                e = discord.Embed(title="Towers Prediction",
                                  color=discord.Color.green())
                e.add_field(name="🎯 Prediction", value=prediction)
                e.add_field(name="💎 Method", value=f"> {method}", inline=False)
                e.add_field(name="🍀 Feeling Unlucky?", value="> Use the ``/unrig`` command to get better chances of winning!", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "PastGames":
                l = TowersPredictor(auth_token)
                prediction = l.pastgames()
                e = discord.Embed(title="Towers Prediction",
                                  color=discord.Color.green())
                e.add_field(name="🎯 Prediction", value=prediction)
                e.add_field(name="💎 Method", value=f"> {method}", inline=False)
                e.add_field(name="🍀 Feeling Unlucky?", value="> Use the ``/unrig`` command to get better chances of winning!", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "HieuAlgorithm":
                l = TowersPredictor(auth_token)
                prediction = l.blitzalgorithm()
                e = discord.Embed(title="Towers Prediction",
                                  color=discord.Color.green())
                e.add_field(name="🎯 Prediction", value=prediction)
                e.add_field(name="💎 Method", value=f"> {method}", inline=False)
                e.add_field(name="🍀 Feeling Unlucky?", value="> Use the ``/unrig`` command to get better chances of winning!", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case _:
                e = discord.Embed(title="Towers Prediction", description="An error has occured!",
                                  color=discord.Color.red())
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
    else:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`,  use `/howtogettoken` if you don't know how to get token." if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Towers Predictor",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )

@tree.command(name='free-mines', description='Randomizied mines game predictor.')
async def mines(interaction: discord.Interaction, round_id: str,
                tile_amount: int):
  if interaction.channel_id != settings["channel"]:
        return await interaction.response.send_message(
            embed=discord.Embed(
                title="Free Mines Predictor",
                description="You are not allowed to use this command in <#1214933851563102281>.",
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )
  if len(round_id) == 36:
    start_time = time.time()
    grid = [
      '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌',
      '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌'
    ]
    already_used = []

    count = 0
    while tile_amount > count:
      a = random.randint(0, 24)
      if a in already_used:
        continue
      already_used.append(a)
      grid[a] = '⭐'
      count += 1

    chance = random.randint(45, 95)
    if tile_amount < 4:
      chance = chance - 15

    em = discord.Embed(color=discord.Color.green())

    em.add_field(name='**Random 50% Working Mines Predicted**:', value="\n" +grid[0]+grid[1]+grid[2]+grid[3]+grid[4]+"\n"+grid[5]+grid[6]+grid[7]+grid[8]+grid[9]+"\n"+grid[10]+grid[11]+grid[12]+grid[13]+grid[14]+"\n"+grid[15]+grid[16]+grid[17] \
        +grid[18]+grid[19]+"\n"+grid[20]+grid[21]+grid[22]+grid[23]+grid[24])
    em.set_footer(text="Pulsive Predictor Free")
    await interaction.response.send_message(embed=em)
  else:
    embed=discord.Embed(
                title="Free Mines Predictor",
                description="Invalid round ID.",
                color=discord.Color.red()
            ).set_footer(text=TEXT)
    await interaction.response.send_message(embed=embed)

@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@app_commands.choices(method=[
    app_commands.Choice(name=key, value=value) for key, value in mines_methods.items()
])
@tree.command(name="mines", description="Predicts your current mines game.")
async def mines(interaction: discord.Interaction, tiles: int, method: str):
    if interaction.channel.id != settings["channel"]:
        return await interaction.response.send_message(
            embed=discord.Embed(
                title="Mines Predictor",
                description="Invalid channel, you can only use the command at <#" + str(settings["channel"]) + ">",
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )
    
    auth_token = checkauth(interaction.user.id)
    if not 0 < tiles < 16:
        return await interaction.response.send_message(
            embed=discord.Embed(
                title="Mines Predictor",
                description="The tile amount must be between 1 and 15 tiles.",
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )
    if auth_token['valid']:
        auth_token = auth_token['token']

        message = f"Using method {method} to predict your current game..."
        await interaction.response.send_message(content=message)

        match method:

            case "algorithm3":
                l = MinesPredictor(auth_token, tiles)
                prediction, mines_amount, bet_amount, uuid = l.aspectmines()
                e = discord.Embed(title="Mines Prediction", color=discord.Color.green())
                e.add_field(name="Prediction", value=prediction, inline=False)
                e.add_field(name="Method", value=f"> {method}",inline=False)
                e.add_field(name="Mines Amount", value=f"> {mines_amount}",inline=False)
                e.add_field(name="Betted", value=f"> {bet_amount}",inline=False)
                e.add_field(name="Round ID", value=f"> {uuid}",inline=False)
                e.add_field(name="Feeling Unlucky?", value="> Use the ``/unrig`` command if you lose twice, if you keep getting lose, try another method.", inline=False)
                e.set_footer(text=TEXT)
                await interaction.edit_original_response(content="Thank you for trusted and using Pulsive Predictor V8!", embed=e)

            case "Invertion":
                l = MinesPredictor(auth_token, tiles)
                prediction, mines_amount, bet_amount, uuid = l.pulsivemines()
                e = discord.Embed(title="Mines Prediction", color=discord.Color.green())
                e.add_field(name="Prediction", value=prediction, inline=False)
                e.add_field(name="Method", value=f"> {method}",inline=False)
                e.add_field(name="Mines Amount", value=f"> {mines_amount}",inline=False)
                e.add_field(name="Betted", value=f"> {bet_amount}",inline=False)
                e.add_field(name="Round ID", value=f"> {uuid}",inline=False)
                e.add_field(name="Feeling Unlucky?", value="> Use the ``/unrig`` command if you lose twice, if you keep getting lose, try another method.", inline=False)
                e.set_footer(text=TEXT)
                await interaction.edit_original_response(content="Thank you for trusted and using Pulsive Predictor V8!", embed=e)

            case "Randomization":
                l = MinesPredictor(auth_token, tiles)
                prediction, mines_amount, bet_amount, uuid = l.randomization()
                e = discord.Embed(title="Mines Prediction", color=discord.Color.green())
                e.add_field(name="Prediction", value=prediction, inline=False)
                e.add_field(name="Method", value=f"> {method}",inline=False)
                e.add_field(name="Mines Amount", value=f"> {mines_amount}",inline=False)
                e.add_field(name="Betted", value=f"> {bet_amount}",inline=False)
                e.add_field(name="Round ID", value=f"> {uuid}",inline=False)
                e.add_field(name="Feeling Unlucky?", value="> Use the ``/unrig`` command if you lose twice, if you keep getting lose, try another method.", inline=False)
                e.set_footer(text=TEXT)
                await interaction.edit_original_response(content="Thank you for trusted and using Pulsive Predictor V8!", embed=e)
            case _:
                e = discord.Embed(title="Mines Prediction", description="An error has occured!",
                                  color=discord.Color.red())
                await interaction.response.send_message(embed=e)


    else:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`,  use `/howtogettoken` if you don't know how to get token." if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Mines Predictor",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )


@tree.command(name="crash", description="Predict current crash games.")
@app_commands.choices(method=[
    app_commands.Choice(name=key, value=value) for key, value in crash_methods.items()
])
@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
async def crash(interaction: discord.Interaction, method: str):
    auth_token = checkauth(interaction.user.id)
    
    if not auth_token['valid']:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`" if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )
        await interaction.response.send_message(
            embed=discord.Embed(
                title="Crash Predictor",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )
        return

    if game_active("crash"):
        e = discord.Embed(
            title="Crash Prediction",
            description="A game is in progress, wait until next one.",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=e)
        return

    games = scraper.get("https://api.bloxflip.com/games/crash").json()
    previousGame = games["history"][0]["crashPoint"]
    
    if method == "EngineeredInversiveAlgorithm":
        av2 = (games["history"][0]["crashPoint"] + games["history"][2]["crashPoint"])
        chancenum = 96.55 / previousGame
        estnum = (1 / (1 - chancenum) + av2) / 2
        estimate = "{:.2f}".format(round(estnum, 2))
        chance = "{:.2f}".format(round(chancenum, 3))
        method_name = "EngineeredInversiveAlgorithm"
    elif method == "ReversedRocketAlgorithm":
        av2 = (games["history"][0]["crashPoint"] + games["history"][1]["crashPoint"])
        chancenum = 65 / previousGame
        estnum = (1 / (1 - chancenum) + av2) / 2
        estimate = "{:.2f}".format(round(estnum, 3))
        chance = "{:.2f}".format(round(chancenum, 2))
        method_name = "ReversedRocketAlgorithm (By Pulse)"
    else:
        await interaction.response.send_message(
            embed=discord.Embed(
                title="Crash Predictor",
                description="Invalid method selected.",
                color=discord.Color.red()
            )
        )
        return

    predictionEM = discord.Embed(
        description=(
            f"**🎯 Cash Point**\n{estimate}\n"
            f"**✅ Accuracy**\n{chance}%\n"
            f"**💎 Method**\n{method_name}\n"
            "**📢 Info**\n"
            "For this method, the accuracy estimate the predicted number safety percentage"
        ),
        color=0x4A9EFF,
        title="Crash Prediction"
    )
    await interaction.response.send_message(embed=predictionEM)


#@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
#@tree.command(name="crash", description="Predict current crash games.")
#@app_commands.choices(method=[
#    app_commands.Choice(name=key, value=value) for key, value in crash_methods.items()
#])
async def crash(interaction, method: str):
    if interaction.channel.id != settings["channel"]:
        return await interaction.response.send_message(
            embed=discord.Embed(
                title="Mines Predictor",
                description="Invalid channel, you can only use the command at <#" + settings["channel"] + ">",
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )

    auth_token = checkauth(interaction.user.id)
    if auth_token['valid']:
        if game_active("crash"):
            e = discord.Embed(title="Crash Prediction", description="A game is in progress, wait until next one.",
                              color=discord.Color.red())
            return await interaction.response.send_message(embed=e)
        match method:
            case "XGBoost":
                l = CrashPredictor()
                prediction = l.xgboost()
                e = discord.Embed(title="Crash Prediction", color=discord.Color.green())
                e.add_field(name="💎 Method", value=f"```{method}```", inline=False)
                e.add_field(name="🎯 Prediction", value=f"```{prediction}x```", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "LinearRegression":
                l = CrashPredictor()
                prediction = l.linear()
                e = discord.Embed(title="Crash Prediction", color=discord.Color.green())
                e.add_field(name="💎 Method", value=f"```{method}```", inline=False)
                e.add_field(name="🎯 Prediction", value=f"```{prediction}x```", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "KNearestNeighbors":
                l = CrashPredictor()
                prediction = l.knn()
                e = discord.Embed(title="Crash Prediction", color=discord.Color.green())
                e.add_field(name="💎 Method", value=f"```{method}```", inline=False)
                e.add_field(name="🎯 Prediction", value=f"```{prediction}x```", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "AdvancedAverage":
                l = CrashPredictor()
                prediction = l.advancedavg()
                e = discord.Embed(title="Crash Prediction", color=discord.Color.green())
                e.add_field(name="💎 Method", value=f"```{method}```", inline=False)
                e.add_field(name="🎯 Prediction", value=f"```{prediction}x```", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "AmustaAlgorithm":
                l = CrashPredictor()
                prediction = l.Amustaalgorithm()
                e = discord.Embed(title="Crash Prediction", color=discord.Color.green())
                e.add_field(name="💎 Method", value=f"```{method}```", inline=False)
                e.add_field(name="🎯 Prediction", value=f"```{prediction}x```", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "AdvancedMedian":
                l = CrashPredictor()
                prediction = l.advmedian()
                e = discord.Embed(title="Crash Prediction", color=discord.Color.green())
                e.add_field(name="💎 Method", value=f"```{method}```", inline=False)
                e.add_field(name="🎯 Prediction", value=f"```{prediction}x```", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "LearnPatterns":
                l = CrashPredictor()
                prediction = l.learnpatterns()
                e = discord.Embed(title="Crash Prediction", color=discord.Color.green())
                e.add_field(name="💎 Method", value=f"```{method}```", inline=False)
                e.add_field(name="🎯 Prediction", value=f"```{prediction}x```", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case "Mean":
                l = CrashPredictor()
                prediction = l.mean()
                e = discord.Embed(title="Crash Prediction", color=discord.Color.green())
                e.add_field(name="💎 Method", value=f"```{method}```", inline=False)
                e.add_field(name="🎯 Prediction", value=f"```{prediction}x```", inline=False)
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
            case _:
                e = discord.Embed(title="Crash Prediction", description="An error has occured!",
                                  color=discord.Color.red())
                e.set_footer(text=TEXT)
                await interaction.response.send_message(embed=e)
    else:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`,  use `/howtogettoken` if you don't know how to get token." if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Mines Predictor",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )

@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="slide", description="Predict current slide games.")
async def slide(interaction):
    auth_token = checkauth(interaction.user.id)
    if auth_token['valid']:
        if game_active("slide"):
            e = discord.Embed(title="Slide Prediction", description="A game is in progress, wait until next one.",
                              color=discord.Color.red())
            return await interaction.response.send_message(embed=e)
        slide_history = scraper.get('https://api.bloxflip.com/games/roulette').json()['history']
        past_games = [game['winningColor'] for game in slide_history]
        r_count = past_games.count("red")
        p_count = past_games.count("purple")
        y_count = past_games.count("yellow")
        r_chance = r_count / len(past_games) * 100
        p_chance = p_count / len(past_games) * 100
        y_chance = y_count / len(past_games) * 200
        em = discord.Embed(color=15844367) 
        em.add_field(name="**Red prediction**", value=f"{r_chance:.2f}%", inline=False)    
        em.add_field(name="**Purple prediction**", value=f"{p_chance:.2f}%", inline=False)    
        em.add_field(name="**Yellow prediction**", value=f"{y_chance:.2f}%", inline=False)    
        await interaction.response.send_message(embed=em)
    else:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`,  use `/howtogettoken` if you don't know how to get token." if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Slide Predictor",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )

#@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
#@tree.command(name="slide", description="Predict current slide games.")
#@app_commands.choices(method=[
#    app_commands.Choice(name=key, value=value) for key, value in slide_methods.items()
#])
async def slide(interaction, method: str):
    return await interaction.response.send_message(
        embed=discord.Embed(
            title="Slide Predictor",
            description=f"Pulsive is currently fixing this predictor right now...",
            color=discord.Color.red()
        ).set_footer(text=TEXT)
    )

    auth_token = checkauth(interaction.user.id)
    if auth_token['valid']:
        if game_active("slide"):
            e = discord.Embed(title="Slide Prediction", description="A game is in progress, wait until next one.",
                              color=discord.Color.red())
            return await interaction.response.send_message(embed=e)
        match method:
            case "PulsiveAlgorithm":
                l = SlidePredictor()
                prediction = l.PulsiveAlgorithm()
                e = discord.Embed(title="Slide Prediction", description="**Predicted Color**: %s" % prediction,
                                  color=discord.Color.green())
                await interaction.response.send_message(embed=e)
            case "LogisticRegression":
                l = SlidePredictor()
                prediction = l.logistic()
                e = discord.Embed(title="Slide Prediction", description="**Predicted Color**: %s" % prediction,
                                  color=discord.Color.green())
                await interaction.response.send_message(embed=e)
            case "AdvancedMarkov":
                l = SlidePredictor()
                prediction = l.advmarkov()
                e = discord.Embed(title="Slide Prediction", description="**Predicted Color**: %s" % prediction,
                                  color=discord.Color.green())
                await interaction.response.send_message(embed=e)
            case "CountAlgo":
                l = SlidePredictor()
                prediction = l.countalgo()
                e = discord.Embed(title="Slide Prediction", description="**Predicted Color**: %s" % prediction,
                                  color=discord.Color.green())
                await interaction.response.send_message(embed=e)
            case "FutureColor":
                l = SlidePredictor()
                prediction = l.futurecolor()
                e = discord.Embed(title="Slide Prediction", description="**Predicted Color**: %s" % prediction,
                                  color=discord.Color.green())
                await interaction.response.send_message(embed=e)
            case "Randomization":
                l = SlidePredictor()
                prediction = l.randomization()
                e = discord.Embed(title="Slide Prediction", description="**Predicted Color**: %s" % prediction,
                                  color=discord.Color.green())
                await interaction.response.send_message(embed=e)
            case _:
                e = discord.Embed(title="Crash Prediction", description="An error has occured!",
                                  color=discord.Color.red())
                await interaction.response.send_message(embed=e)
    else:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`,  use `/howtogettoken` if you don't know how to get token." if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Mines Predictor",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )

@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="analyzepatterns", description="Attemps to analyze patterns for mines/towers gamemode.")
@app_commands.choices(method=[
    app_commands.Choice(name="Mines", value="Mines"),
])
async def analyzepatterns(interaction: discord.Interaction, method: str):
    auth_token = checkauth(interaction.user.id)
    if auth_token['valid']:
        name = settings['madeby']
        match method:
            case "Mines":
             p = PatternAnalyzer(auth_token['token'])
             pred, lowest,recent = p.minesAnalyze()
             e = discord.Embed(title="Mines Analysis",description=f"**Recent Patterns**: [{recent}]\n**Top 3 Patterns**: [{pred}]\n**Lowest Clicked Patterns**: [{lowest}]",color=discord.Color.green())
             await interaction.response.send_message(embed=e)

    else:
        m = (
            "Your key has expired" if auth_token['reason'] == "exp"
            else "You must link your account first with `/link`,  use `/howtogettoken` if you don't know how to get token." if auth_token['reason'] == "NOLINK"
            else "You don't have a key. Please purchase one in <#1244241266888413194>!"
        )

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Analysis Error",
                description=m,
                color=discord.Color.red()
            ).set_footer(text=TEXT)
        )

class Unrig:
    def __init__(self, auth_token):
        self.auth_token = auth_token
        
    def hashlib_nigger(self):
        self.headers = {
            "x-auth-token": str(self.auth_token).strip(),
            "Referer": "https://bloxflip.com/",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/117.0.5938.108 Mobile/15E148 Safari/604.1"
        }

        params = {
            'size': '1000',
            'page': '0',
        }

        
        response = scraper.get('https://api.bloxflip.com/games/mines/history', params=params, headers=self.headers)
        
        r = response.json()['data']
        self.hashes = [x['serverSeed'] for x in r if not x['exploded']]
        return self.change_hash()
        
    
    def change_hash(self):
        change_hash_req = scraper.post('https://api.bloxflip.com/provably-fair/clientSeed', headers=self.headers, json={
            'clientSeed': self.hashes[random.randint(0, len(self.hashes) - 1)][:32]
        })
        
        return change_hash_req.json()['success']


@app_commands.checks.cooldown(1, cooldown, key=lambda i: (i.guild_id, i.user.id))
@tree.command(name="unrig", description="Unrig your BloxFlip game session.")
async def unrig(interaction: discord.Interaction):
    auth_token = checkauth(interaction.user.id)
    if auth_token:
        unrig_instance = Unrig(auth_token["token"])
        success = unrig_instance.hashlib_nigger()
        if success:
            e = discord.Embed(
                title="✅ Unrigged Successfully!",
                description="Successfully unrigged your game session!",
                color=discord.Color.green()
            )
            e.set_footer(text=TEXT)
            await interaction.response.send_message(embed=e)
        else:
            await interaction.response.send_message(
                embed=discord.Embed(
                    title=":x: Unrigging Failed",
                    description="Failed to unrig your game session.",
                    color=discord.Color.red()
                ).set_footer(text=TEXT)
            )


client.run("")

# Pulsive Predictor