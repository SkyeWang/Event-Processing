使用前需要下载edgedriver，在edge上登陆twitter，然后改三个地方：
151行     user_data_dir = r"C:\Users\30751\AppData\Local\Microsoft\Edge\User Data"   
（一般只有用户名不同）
172行     options.binary_location = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
（一般不用改）
174行     edge_service = EdgeService(executable_path=r"D:\edgedriver\msedgedriver.exe")
（根据保存edgedriver的地址改）
关键词按照keywords里那样就行，可以让ai改
