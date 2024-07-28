import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV


def display(test, train, mse, r2):
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print(f"Test score: {test}")
    print(f"Train score: {train}")

def display_tuning(test,train,val,param,score):
    print(f"Best params: {param}")
    print(f"Best scores: {score}")
    print(f"Test score: {test}")
    print(f"Train score: {train}")
    print(f"Validation score: {val}")

def evaluation_model(X, y, clf, n_runs=30, test_size=0.2):
    test_score = []
    train_score = []
    mse = []
    r2 = []
    for i in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = clf.score(X_test, y_test)
        score2 = clf.score(X_train, y_train)
        test_score.append(score)
        train_score.append(score2)
        mse.append(mean_squared_error(y_test, y_pred))
        r2.append(r2_score(y_test, y_pred))

    display(np.mean(test_score), np.mean(train_score), np.mean(mse), np.mean(r2))
    return np.mean(test_score), np.mean(train_score), np.mean(mse), np.mean(r2)


def evaluation_model_unbias(df, input_features, output_feature, team_ids, clf, n_runs=30, test_size=0.2):
    test_score = []
    train_score = []
    mse = []
    r2 = []
    for i in range(n_runs):
        train_teams, test_teams = train_test_split(team_ids, test_size=test_size, random_state=i)
        train_df = df[df['Team ID'].isin(train_teams)]
        test_df = df[df['Team ID'].isin(test_teams)]

        X_train = train_df[input_features]
        X_test = test_df[input_features]
        y_train = train_df[output_feature]
        y_test = test_df[output_feature]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = clf.score(X_test, y_test)
        score2 = clf.score(X_train, y_train)
        test_score.append(score)
        train_score.append(score2)
        mse.append(mean_squared_error(y_test, y_pred))
        r2.append(r2_score(y_test, y_pred))

    display(np.mean(test_score), np.mean(train_score), np.mean(mse), np.mean(r2))
    return np.mean(test_score), np.mean(train_score), np.mean(mse), np.mean(r2)


def hyperparameter_tuning_random(X_train, X_test, X_val, y_train, y_test, y_val, param_grid, clf, n_splits=10, n_repeats=3):
    best_scores = []
    best_params = []
    val_score = []
    test_score = []
    train_score = []
    for i in range(n_repeats):
        clf_random = RandomizedSearchCV(estimator=clf, param_distributions=param_grid,
                                        n_iter=100, cv=n_splits, verbose=0, random_state=42, n_jobs=-1,
                                        scoring='neg_mean_squared_error')
        clf_random.fit(X_val, y_val)
        best_score = clf_random.best_score_
        best_scores.append(best_score)
        best_params.append(clf_random.best_params_)
        # mse = mean_squared_error(y_test_s, y_pred)
        # rmse = mse ** 0.5
        best_params = clf_random.best_params_
        best_model = clf_random.best_estimator_

        val_score.append(best_model.score(X_val, y_val))
        # Evaluate the best model on the test set
        test_score.append(best_model.score(X_test, y_test))
        train_score.append(best_model.score(X_train, y_train))

    display_tuning(np.mean(test_score), np.mean(train_score), np.mean(val_score), best_params)
    return np.mean(test_score), np.mean(train_score), np.mean(val_score), best_params, np.mean(best_scores)

def hyperparameter_tuning_grid(X_train, y_train, X_val, y_val, X_test, y_test, param_grid, clf, n_repeats=3, n_splits=10):
    best_scores = []
    best_params = []
    val_score = []
    test_score = []
    train_score = []
    for i in range(n_repeats):
        grid_search = GridSearchCV(clf, param_grid, cv=n_splits, scoring='r2')
        grid_search.fit(X_val, y_val)

        best_score = grid_search.best_score_
        best_scores.append(best_score)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        val_score.append(best_model.score(X_val, y_val))
        # Evaluate the best model on the test set
        test_score.append(best_model.score(X_test, y_test))
        train_score.append(best_model.score(X_train, y_train))

    display_tuning(np.mean(test_score), np.mean(train_score), np.mean(val_score), best_params)
    return np.mean(test_score), np.mean(train_score), np.mean(val_score), best_params, np.mean(best_scores)