from sklearn.preprocessing import StandardScaler

def feature_engineering(df):
    df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                             (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['PromoOpen'] = 12 * (df['Year'] - df['Promo2SinceYear']) + \
                      (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
    df.fillna(0, inplace=True)
    return df

def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler
