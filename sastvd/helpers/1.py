# import csv
# if __name__ =='__main__':
#     filePath="/opt/linevd-main/storage/external/MSR_data_cleaned.csv"
#
#     with open(filePath, 'r',encoding="utf-8") as csvfile:
#         reader = csv.reader(csvfile)
#         # print(len())
#         count=0
#         f = open('/opt/linevd-main/storage/external/MSR_data_cleaned_mi.csv', 'w', encoding='utf-8',newline="")
#         csv_write = csv.writer(f)
#         for row in reader:
#             #the first row is table header
#             print(row)
#             #type:# list
#             # print(type(row))
#             # data=row
#
#             # if count%100==0:
#             #    csv_write.writerow(row)
#             # count=count+1
#             # print(count)
#             # if count==5000000:
#          #     break
#     print(count)
# from turtle import pd
#
# df = pd.read_csv('.csv',chunksize=100000)
# from multiprocessing import Pool
#
# def f(row):
#     print(row**2)
#
# with Pool(processes=4) as pool:
#
#     # print "[0, 1, 4,..., 81]", 循序返回
#     print(pool.map(f, [1,2,3]))

    # print same numbers in arbitrary order
    # for i in pool.imap_unordered(f, range(10)):
    #     print(i)
# import  pandas as pd
# df=pd.read_parquet("/opt/linved/storage/cache/minimal_datasets/minimal_bigvul_False.pq")
# def cal(row):
#     # if(len(row['diff'])>0):
#         # print(row['id'])
# df.apply(cal,axis=1)
import subprocess
# print( subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT))
# import subprocess
# import os
# import os
# os.environ["JAVA_HOME"] = "/usr/local/jdk-17.0.7"

# def main():
#     python_version = subprocess.check_output(["python", "--version"])
#     java_version = subprocess.check_output(["java", "--version"])
#
#     print(python_version)
#     print(java_version)
#
#     # raw_input()  # equivalent to your pause call
#
#
# if __name__ == '__main__':
#     main()
# import operator
f = open("/home/su/myfile.txt","r")
file_handle=open('/home/su/test.txt',mode='w')
lines = f.readlines()#读取全部内容
for line in lines:
    if operator.contains(str(line),'json'):
        # print(line)
        pass
    else:
        file_handle.write(line+'\n')
# import os
# # n = os.system(commands)
# cmd="/opt/joern/joern-cli/joern --script /opt/linved/storage/external/get_func_graph.scala --params='filename=/opt/linved/storage/processed/bigvul/before/43529.c'"
# # subprocess.run(cmd)
# # subprocess.run(cmd,  stdin=None, input=None, stdout=None, stderr=None, shell=False, timeout=None, check=False, universal_newlines=False)
# subprocess.run(cmd, shell=False)
# print(n)
# import commands
#
# print(commands.getstatus("/home/su/1.bash"))