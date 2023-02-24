import csv
import pandas as pd
kp = [1,2,5,0,14,16,15,17]
classification = {0:"正坐",1:"右歪头",2:"左歪头",3:"右抬肩",4:"左抬肩",5:"低头",6:"仰头"}

data = {0:(1,561),1:(562,807),2:(819,1026),3:(1119,1335),4:(1431,1680),5:(1731,2012),6:(2098,2225)}
print(classification[4])
def get_label(x,data):
    for i in range(7):
        if x >= data[i][0] and x <= data[i][1]:
            return i
    return 0



header = ["1x","1y","2x","2y","5x","5y","0x","0y","14x","14y","16x","16y","15x","15y","17x","17y","class"]
print(header)
dataf=pd.DataFrame(columns=header, index=[])
i = 1
with open("data.txt",'r') as f:
    lines = f.readlines()
    for line in lines:
        
        line = line.replace('\n', '')
        x = line.split(', ')
        x = list(map(int, x))
        temp = x.copy()
        class_x = get_label(i,data)
        x.append(class_x)
        tempData = pd.Series(x, index=header)
        dataf = dataf.append(tempData,ignore_index=True)

        temp_x = [(i + 20)*(i>0) for i in temp]
        temp_x.append(class_x)
        temp_xp = pd.Series(temp_x,index=header)
        dataf = dataf.append(temp_xp,ignore_index=True)

        temp_x = [(i - 20)*(i>0) for i in temp  ]
        temp_x.append(class_x)
        temp_xp = pd.Series(temp_x,index=header)
        dataf = dataf.append(temp_xp,ignore_index=True)

        temp_x= [(i + 20*(temp.index(i)%2))*(i>0) for i in temp]
        temp_x.append(class_x)
        temp_xp = pd.Series(temp_x,index=header)
        dataf = dataf.append(temp_xp,ignore_index=True)

        temp_x= [(i - 20*(temp.index(i)%2))*(i>0) for i in temp  ]
        temp_x.append(class_x)
        temp_xp = pd.Series(temp_x,index=header)
        dataf = dataf.append(temp_xp,ignore_index=True)

        i = i + 1

dataf = dataf.sample(frac=1.0) # 随机打乱数据
dataf.to_csv("data.csv")
        
print(dataf.head())
        

        
