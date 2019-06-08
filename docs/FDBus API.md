# FDBus  API

## Client, Server消息收发相关方法调用

**方法 (CFdbBaseObject, CBaseServer或CBaseClient)**

```c++
void invoke(FdbSessionId_t receiver
                , FdbMsgCode_t code
                , const CFdbBasePayload &data
                , int32_t timeout = 0);

void invoke(FdbSessionId_t receiver
                , FdbMsgCode_t code
                , const void *buffer = 0
                , int32_t size = 0
                , int32_t timeout = 0);
```

-   同步/异步：异步 (同步：调用后阻塞，直至收到对方回复；异步：调用后立刻返回。对方的回复通过回调函数获得)
-   调用者：CBaseServer/CBaseClient；但通常是CBaseServer在回调函数中调用。参数receiver可以从CBaseServer回调函数的参数msg_ref获得。
-   响应者：CBaseServer/CBaseClient；但通常是CBaseClient
-   接收方返回：有返回
-   消息私有数据：默认使用CBaseMessage；不能携带私有数据(所谓消息私有数据是指在发送请求时可以携带额外数据；这些数据在请求返(onReply)时可以获取到。)

```c++
void invoke(FdbSessionId_t receiver
                , CFdbMessage *msg
                , const CFdbBasePayload &data
                , int32_t timeout = 0);

void invoke(FdbSessionId_t receiver
                , CFdbMessage *msg
                , const void *buffer = 0
                , int32_t size = 0
                , int32_t timeout = 0);
```

-   同步/异步：异步
-   调用者：CBaseServer/CBaseClient；但通常是CBaseServer在回调函数中调用。参数receiver可以从CBaseServer回调函数的参数msg_ref获得。
-   响应者：CBaseServer/CBaseClient；但通常是CBaseClient
-   接收方返回：有返回
-   消息私有数据：参数msg扩展CBaseMessage后，可以携带私有数据

```c++
void invoke(FdbMsgCode_t code
                , const CFdbBasePayload &data
                , int32_t timeout = 0);

void invoke(FdbMsgCode_t code
                , const void *buffer = 0
                , int32_t size = 0
                , int32_t timeout = 0);
```

-   同步/异步：异步
-   调用者：CbaseClient
-   响应者：CbaseServer
-   接收方返回：有返回
-   消息私有数据：默认使用CBaseMessage；不能携带私有数据

```c++
void invoke(CFdbMessage *msg
                , const CFdbBasePayload &data
                , int32_t timeout = 0);

void invoke(CFdbMessage *msg
                , const void *buffer = 0
                , int32_t size = 0
                , int32_t timeout = 0);
```

-   同步/异步：异步
-   调用者：CBaseClient
-   响应者：CBaseServer
-   接收方返回：有返回
-   消息私有数据：参数msg扩展CBaseMessage后，可以携带私有数据

```c++
void invoke(FdbSessionId_t receiver
                , CBaseJob::Ptr &msg_ref
                , const CFdbBasePayload &data
                , int32_t timeout = 0);

void invoke(FdbSessionId_t receiver
                , CBaseJob::Ptr &msg_ref
                , const void *buffer = 0
                , int32_t size = 0
                , int32_t timeout = 0);
```

-   同步/异步：同步
-   调用者：CBaseServer/CBaseClient；但通常是CBaseServer
-   响应者：CBaseServer/CBaseClient；但通常是CBaseClient
-   接收方返回：有返回
-   消息私有数据：msg_ref指向CBaseMessage子类对象，该对象可以携带私有数据；如果无私有数据则msg_ref直接指向CBaseMessage

```c++
void invoke(CBaseJob::Ptr &msg_ref
                , const CFdbBasePayload &data
                , int32_t timeout = 0);

void invoke(CBaseJob::Ptr &msg_ref
                , const void *buffer = 0
                , int32_t size = 0
                , int32_t timeout = 0);
```

-   同步/异步：同步
-   调用者：CBaseClient
-   响应者：CBaseServer
-   接收方返回：有返回
-   消息私有数据：msg_ref指向CBaseMessage子类对象，该对象可以携带私有数据；如果无私有数据则msg_ref直接指向CBaseMessage

```c++
void send(FdbSessionId_t receiver
              , FdbMsgCode_t code
              , const CFdbBasePayload &data);

void send(FdbSessionId_t receiver
              , FdbMsgCode_t code
              , const void *buffer = 0
              , int32_t size = 0);
```

-   同步/异步：异步
-   调用者：CBaseServer/CBaseClient；但通常是CBaseServer
-   响应者：CBaseServer/CBaseClient；但通常是CBaseClient
-   接收方返回：无返回
-   消息私有数据：无需携带私有数据

```c++
void send(FdbMsgCode_t code
              , const CFdbBasePayload &data);

void send(FdbMsgCode_t code
              , const void *buffer = 0
              , int32_t size = 0);
```

-   同步/异步：异步
-   调用者：CBaseClient
-   响应者：CBaseServer
-   接收方返回：无返回
-   消息私有数据：无需携带私有数据

```c++
void broadcast(FdbMsgCode_t code
                   , const CFdbBasePayload &data
                   , const char *filter = 0);
void broadcast(FdbMsgCode_t code
                   , const char *filter = 0
                   , const void *buffer = 0
                   , int32_t size = 0);
```

-   同步/异步：N/A
-   调用者：CBaseServer
-   响应者：CBaseClient
-   接收方返回：无返回
-   消息私有数据：无需携带私有数据

```c++
void subscribe(NFdbBase::FdbMsgSubscribe &msg_list
                   , int32_t timeout = 0);
```

-   同步/异步：异步
-   调用者：CBaseClient
-   响应者：CBaseServer
-   接收方返回：有返回
-   消息私有数据：默认使用CBaseMessage；不能携带私有数据

```c++
void subscribe(NFdbBase::FdbMsgSubscribe &msg_list
                   , CFdbMessage *msg
                   , int32_t timeout = 0);
```

-   同步/异步：异步
-   调用者：CBaseClient
-   响应者：CBaseServer
-   接收方返回：有返回
-   消息私有数据：参数msg扩展CBaseMessage后，可以携带私有数据

```c++
void subscribe(CBaseJob::Ptr &msg_ref
                   , NFdbBase::FdbMsgSubscribe &msg_list
                   , int32_t timeout = 0);
```

-   同步/异步：同步
-   调用者：CBaseClient
-   响应者：CBaseServer
-   接收方返回：有返回
-   消息私有数据：msg_ref指向CBaseMessage子类对象，该对象可以携带私有数据；此处msg_ref也可以不指向任何对象，即不携带私有数据。

```c++
void unsubscribe(NFdbBase::FdbMsgSubscribe &msg_list);
```

-   同步/异步：异步
-   调用者：CBaseClient
-   响应者：CBaseServer
-   接收方返回：无返回
-   消息私有数据：无需携带私有数据

## Client, Server其它方法调用

### 类CBaseClient

```C++
CBaseClient(const char *name, CBaseWorker *worker = 0)
```

-   描述：构造函数；创建Client
-   参数
    -   name : client名字，可以为任意字符串
    -   worker：如果指定则所有回调函数在该worker上运行；否则在fdbus工作线程FDB_CONTEXT上运行

```c++
FdbSessionId_t connect(const char *url = 0)
```

-   描述：连接到Server上
-   参数：
    -   url：Server的地址，支持tcp，file和svc格式。如果不指定默认是svc://name，name是构造函数的name参数。

```C++
void disconnect(FdbSessionId_t sid = FDB_INVALID_ID);
```

-   描述：断开连接
-   参数
    -   sid：通常不指定

### 类CBaseServer

```C++
CBaseServer(const char *name, CBaseWorker *worker = 0);
```

-   描述：构造函数；创建Server
-   参数
    -   name : server名字，可以为任意字符串
    -   worker：如果指定则所有回调函数在该worker上运行；否则在fdbus工作线程FDB_CONTEXT上运行

```C++
FdbSocketId_t bind(const char *url = 0);
```

-   描述：绑定地址
-   参数
    -   url：Server的地址，支持tcp，file和svc格式。如果不指定默认是svc://name，name是构造函数的name参数。

```C++
void unbind(FdbSocketId_t skid = FDB_INVALID_ID);
```

-   描述：解除地址绑定
-   参数
    -   sid：通常不指定

### 类CFdbBaseObject

```C++
CFdbBaseObject(const char *name = 0, 
               CBaseWorker *worker = 0, 
               EFdbEndpointRole role = FDB_OBJECT_ROLE_UNKNOWN)
```

-   描述：构造函数；创建对象
-   参数
    -   name : object名字，可以为任意字符串
    -   worker：如果指定则所有回调函数在该worker上运行；否则在fdbus工作线程FDB_CONTEXT上运行
    -   role：通常不指定

```C++
FdbObjectId_t bind(CBaseEndpoint *endpoint, 
                   FdbObjectId_t oid = FDB_INVALID_ID)
```

-   描述：绑定对象ID，将对象角色设置为Server。
-   参数
    -   endpoint：该对象要要挂靠到哪个endpoint上，可以是CBaseServer，也可以是CBaseClient
    -   oid：需要绑定的对象id，范围是1~65535。如果不指定则系统自动分配


```C++
FdbObjectId_t connect(CBaseEndpoint *endpoint, 
                      FdbObjectId_t oid = FDB_INVALID_ID)
```

-   描述：绑定对象ID，将对象角色设置为Client。
-   参数
    -   endpoint：该对象要要挂靠到哪个endpoint上，可以是CBaseServer，也可以是CBaseClient
    -   oid：需要绑定的对象id，范围是1~65535。如果不指定则系统自动分配


```C++
void unbind()
```

-   描述：解除Server对象的绑定

```C++
void disconnect()
```

-   描述：解除Client对象的绑定

### 类FDB_CONTEXT

```c++
bool start(uint32_t flag = FDB_WORKER_ENABLE_FD_LOOP)
```

-   描述：启动fdbus收发线程；其本质就是一个worker线程
-   参数
    -   flag：启动标志；由以下标志位组合而成
        -   FDB_WORKER_EXE_IN_PLACE：置位表示不启动线程，在调用的地方运行worker的主循环；否则启动新线程执行worker的主循环

## Client, Server的回调函数

**回调函数 (CFdbBaseObject, CBaseServer或CBaseClient)**

```c++
virtual void onOnline(FdbSessionId_t sid, bool is_first)
```

-   触发条件
    -   CBaseServer：当有Client连上时被调用到；is_first表示是第一个Client来连接
    -   CBaseClient：当和Server连上时被调用到；由于client不会连接多个server，is_first总是true
    -   CBaseObject：作为Server时，调用时机和CBaseServer一致；作为Client时，调用时机和CBaseClient一致


```c++
virtual void onOffline(FdbSessionId_t sid, bool is_last)
```

-   触发条件
    -   CBaseServer：当有Client断开连接时被调用到；is_last表示最后一个client断开连接
    -   CBaseClient：当和Server断开连接时被调用到；由于client不会连接多个server，is_last总是true
    -   CBaseObject：作为Server时，调用时机和CBaseServer一致；作为Client时，调用时机和CBaseClient一致


```c++
virtual void onInvoke(CBaseJob::Ptr &msg_ref)
```

-   触发条件
    -   CBaseServer：当相连的Client调用CBaseClient::invoke()时被调用到；
    -   CBaseClient通常不需要实现这个方法
    -   CBaseObject：作为Server时，调用时机和CBaseServer一致


```c++
virtual void onReply(CBaseJob::Ptr &msg_ref)
```

-   触发条件
    -   CBaseServer通常不需要实现这个方法；
    -   CBaseClient：当相连的Server调用CBaseMessage::reply()时被调到
    -   CBaseObject：作为Client时，调用时机和CBaseClient一致


```c++
virtual void onBroadcast(CBaseJob::Ptr &msg_ref)
```

-   触发条件
    -   CBaseServer通常不需要实现这个方法；
    -   CBaseClient：当相连的Server调用CBaseServer::broadcast()时被调到
    -   CBaseObject：作为Client时，调用时机和CBaseClient一致


```c++
virtual void onSubscribe(CBaseJob::Ptr &msg_ref)
```

-   触发条件
    -   CBaseServer：当有client调用CBaseClient::subscribe()时被调用到；该函数必须调用
    -   CBaseMessage::broadcast()将被注册事件的初始值返回给发起注册请求的Client；
    -   CBaseClient通常不需要实现这个方法。
    -   CBaseObject：作为Server时，调用时机和CBaseServer一致


```c++
void onCreateObject(CBaseEndpoint *endpoint, CFdbMessage *msg)
```

-   触发条件
    -   CBaseServer/CBaseClient：当对方和object通信，但object尚未创建时，该方法被调用到。该方法必须根据msg里的内容创建指定的CBaseObject并bind/connect。
    -   CBaseObject：无此回调


```c++
bool connectionEnabled(const NFdbBase::FdbMsgAddressList &addr_list)
```

-   触发条件
    -   CBaseServer：无此回调
    -   CBaseClient：当server上线时，name
        Server会把Server相关的信息广播给监听该server的client。addr_list包含了Server相关的信息，例如Server所在的host，Server地址等等。在连接Server之前先调用该回调，如果返回true则会连接server，否则将不会连接到Server上。注意，无论是否指定工作线程，该方法只在FDBus工作线程FDB_CONTEXT上执行。

## Message方法调换

**方法 (CBaseMessage或CFdbMessage)**

```c++
static void reply(CBaseJob::Ptr &msg_ref, 
                  const CFdbBasePayload &data);
```

-   描述：Server端收到Client发送过来的Message后，通过该函数将Protobuf格式的数据包返回给该Client

```c++
static void reply(CBaseJob::Ptr &msg_ref, 
                  const void *buffer = 0, 
                  int32_t size = 0);
```

-   描述：同上，只是返回的是raw data


```c++
bool deserialize(CFdbBasePayload &payload)
```

-   描述：将收到的protobuf格式的流反序列化成C++数据结构


```c++
void *getPayloadBuffer()
```

-   描述：获得消息中payload数据的起始地址。对于protobuf格式的payload，通过调用deserialize()可以获得消息数据，无需调用该函数。该函数用于raw data的消息


```c++
bool isRawData()
```

-   描述：如果消息数据是protobuf则返回false；否则返回true


```c++
FdbMsgCode_t code()
```

-   描述：获得消息ID


```c++
virtual const char *getFilter()
```

-   描述：用于onBroadcast()，获得广播过来的消息的filter


```c++
FdbSessionId_t session()
```

-   描述：获得消息的session


```c++
bool isError()
```

-   描述：用在onReply()里，检查对方返回的消息里是否置上了错误标记；如果置上错误标记说明这次请求发生了错误。此时需要调用decodeStatus()获得错误代码和错误描述从而做进一步检查


```c++
bool isStatus()
```

-   描述：为true说明返回的消息不包含实际数据，而是个状态，表明请求是成功处理了还是发生其它情况，例如请求超时。isError()是isStatus()的一个特例。


```c++
bool decodeStatus(int32_t &error_code, 
                  std::string &description)
```

-   描述：当isStatus()或isError()为true时，获得状态的ID和描述。


```c++
void status(CBaseJob::Ptr &msg_ref, 
            int32_t error_code, 
            const char *description = 0);
```

-   描述：和reply()类似，只是返回的不是数据，而是状态。通常用于没有数据返回的请求。


```c++
void broadcast(FdbMsgCode_t code
                   , const CFdbBasePayload &data
                   , const char *filter = 0);
void broadcast(FdbMsgCode_t code
                   , const char *filter = 0
                   , const void *buffer = 0
                   , int32_t size = 0);
```

-   描述：在CBaseServer的onBroadcast调用，将所注册消息的初始值返回给前来注册的client端。

## Protobuf

```protobuf
package NFdbBase;

message FdbMsgHostAddress
{
    required string ip_address = 1;
    required string ns_url = 2;
    optional string host_name = 3;
}
```

-   Generated C++ Class：`::NFdbBase::FdbMsgHostAddress`

-   Retrieve Memebers

    -   ```c++
        ::NFdbBase::FdbMsgHostAddress::ip_address()
        ::NFdbBase::FdbMsgHostAddress::ns_url()
        ::NFdbBase::FdbMsgHostAddress::host_name()
        
        对于可选项host_name，还有成员函数::NFdbBase::FdbMsgHostAddress::has_host_name()用于检查该成员是否设置。
        ```

-   Change Members

    -   ```c++
        ::NFdbBase::FdbMsgHostAddress::set_ip_address()
        ::NFdbBase::FdbMsgHostAddress::set_ns_url()
        ::NFdbBase::FdbMsgHostAddress::set_host_name()
        ```

```protobuf
package NFdbBase;

message FdbMsgHostAddressList
{
    repeated FdbMsgHostAddress address_list = 1;
}
```

-   Generated C++ Class：`::NFdbBase::FdbMsgHostAddressList`

-   Retrieve Memebers

    -   ```c++
        NFdbBase::FdbMsgHostAddressList host_list;
        
        const ::google::protobuf::RepeatedPtrField< ::NFdbBase::FdbMsgHostAddress> &addr_list = host_list.address_list();
        
        for (::google::protobuf::RepeatedPtrField< ::NFdbBase::FdbMsgHostAddress>::const_iterator it = addr_list.begin(); it != addr_list.end(); ++it)
        {
            const ::NFdbBase::FdbMsgHostAddress &addr = *it;
            std::cout << ""ip: "" << addr.ip_address() << "", url: "" << addr.ns_url() << "", name: "" << addr.host_name();
        }
        ```

-   Change Members

    -   ```c++
        ::NFdbBase::FdbMsgHostAddressList &host_tbl;
        ::NFdbBase::FdbMsgHostAddress *addr = host_tbl.add_address_list();
        addr.set_ip_address(""192.168.0.1"");
        addr.set_ns_url(""http://server"");
        addr.set_host_name(""file server"");
        ```

