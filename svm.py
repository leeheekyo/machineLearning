from sklearn.svm import SVR

x = [[0, 0, 0], [1, 1, 1]]               
y = [0, 1]

svr = SVR()
svr.fit(x, y)

ret = svr.predict([[1, 1, 0]])

print(ret)