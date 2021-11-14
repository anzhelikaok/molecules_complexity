import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


class MoleculesComplexPredictor:

    def train(self, mol_data_file):
        df = pd.read_csv(mol_data_file, delimiter=';')

        X_train = df.values[:, :-1]
        Y_train = df.values[:, -1]
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        self.lr = LinearRegression()
        self.lr.fit(X_train_scaled, Y_train)

        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        rf.fit(X_train_scaled, Y_train)

        return df.columns[:-1], rf.feature_importances_

    def predict(self, input):
        x_test_scaled = self.scaler.transform(input)
        return self.lr.predict(x_test_scaled)


mcp = MoleculesComplexPredictor()
columns, fi = mcp.train("mol_data.csv")

plt.barh(columns, fi)
plt.savefig('rf_importance.png')
plt.clf()