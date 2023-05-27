import datetime
import pymysql
 
# 打开数据库连接
db = pymysql.connect(host='localhost',
                     user='root',
                     password='1216',
                     database='bysj')
 
# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()
 


## 插入
# sql = "INSERT  INTO  posedata (`日期`,`右歪头`,`左歪头`,`右抬肩`,`左抬肩`,`低头`,`仰头`,`时长`) VALUES (\
#                         DATE('2023-5-2'),  20,      30,     40,     50,    60,     70,  80)"
# try:
#     cursor.execute(sql)
#     db.commit()
# except:
#     print("failed")
# cursor.execute("SELECT `日期` FROM POSEDATA")

## 删除
# sql = "DELETE FROM posedata WHERE `日期`=DATE('2023-05-02')"
# try:
#     cursor.execute(sql)
#     db.commit()
# except:
#     print("failed")
# cursor.execute("SELECT `日期` FROM POSEDATA")

## 查询
sql = "SELECT * FROM posedata WHERE `日期`=DATE('2023-04-29')"
try:
    cursor.execute(sql)
    data = cursor.fetchall()
    print(data)
except:
    print("failed")
# 关闭数据库连接
db.close()