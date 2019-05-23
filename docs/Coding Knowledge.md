# 重点内容

## 基础算法

### 递归

#### 汉诺塔

```c++
# include <iostream>

using namespace std;

/**

-   打印 Hanoi塔移动过程
-   @param n 塔的层数
-   @param from 塔的开始位置
-   @param to 塔的结束位置
-   @param help 中介
*/
void printHanoiTower(const int n, const string &from, const string &to, const string &help) {
if (n == 1) // 递归终止
    cout << "move " << n << " from " << from << " to " << to << endl;
else {
    printHanoiTower(n - 1, from, help, to); // 移动上面的 n - 1 层
    cout << "move " << n << " from " << from << " to " << to << endl; // 移动最下面一层
    printHanoiTower(n - 1, help, to, from); // 将上面的 n - 1 层移回
}
}

int main() {
    int n;
    cin >> n;
    string from = "A";
    string to = "B";
    string help = "C";
    printHanoiTower(n, from, to, help);
    return 0;
}
```

### 排序

#### 快排

```c++
// 快速排序各类实现

// 快速排序.三指针分区法.cpp

// 快速排序 三指针分区法
// 类比单向扫描法，增加的是对小于、等于两种情况的区分

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * 快速排序
 * @param nums 被排序数组
 * @param low 首个元素索引
 * @param high 末尾元素索引
 */
void quickSort(vector<int> &nums, int low, int high) {
    if (low < high) { // 不要忽略该判断，归并排序中也有类似判断
        int pivot = nums[low];
        int lt = low; // less than 指向首个不小于 pivot 的元素
        int i = low + 1; // pointer
        int gt = high; // greater than 指向最后一个不大于 pivot 的元素

        while (i <= gt) {
            if (nums[i] < pivot)
                swap(nums[i++], nums[lt++]);
            else if (nums[i] > pivot)
                swap(nums[i], nums[gt--]);
            else i++;
        }

        // lt 到 gt 之间的数都等于 pivot，不必再排序
        quickSort(nums, low, lt - 1);
        quickSort(nums, gt + 1, high);
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    int n;
    cin >> n;

    vector<int> nums(n);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    quickSort(nums, 0, n - 1); // 后两个参数是：首个元素的索引和末尾元素的索引

    for (int i = 0; i < n; i++) {
        cout << nums[i] << " ";
    }
    return 0;
}
// 快速排序.单向扫描分区法.cpp

// 快速排序 一遍单向扫描法

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * 将数组按 pivot 切分
 * @param nums 被切分数组，nums[high]为pivot
 * @param low 首个元素索引
 * @param high 末尾元素索引
 * @return pivot 索引
 */
int partition(vector<int> &nums, int low, int high) {
    int pivot = nums[high]; // 以最后一个元素为 pivot
    int i = low - 1; // i 及其前面的元素小于等于 pivot
    for (int j = low; j < high; j++) { // 注意在循环过程中，pivot 不应被交换
        if (nums[j] <= pivot)
            swap(nums[++i], nums[j]);
    }
    swap(nums[high], nums[i + 1]); // 交换 pivot 到合适位置
    return i + 1;
}

/**
 * 快速排序
 * @param nums 被排序数组
 * @param low 首个元素索引
 * @param high 末尾元素索引
 */
void quickSort(vector<int> &nums, int low, int high) {
    if (low < high) { // 不要忽略该判断，归并排序中也有类似判断
        int par = partition(nums, low, high);
        quickSort(nums, low, par - 1);
        quickSort(nums, par + 1, high);
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    int n;
    cin >> n;

    vector<int> nums(n);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    quickSort(nums, 0, n - 1); // 后两个参数是：首个元素的索引和末尾元素的索引

    for (int i = 0; i < n; i++) {
        cout << nums[i] << " ";
    }
    return 0;
}

// 快速排序.双向扫描分区法.cpp

// 快速排序

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * 将数组按 pivot 切分
 * @param nums 被切分数组，nums[high]为pivot
 * @param low 首个元素索引
 * @param high 末尾元素索引
 * @return pivot 索引
 */
int partition(vector<int> &nums, int low, int high) {
    int pivot = nums[low];
    int left = low + 1;
    int right = high;
    while (left <= right) { // 必须取等
        while (left <= right && nums[left] <= pivot) left++; // left 指向首个大于 pivot 的元素
        while (left <= right && nums[right] > pivot) right--; // right 指向最后一个小于等于 pivot 的元素
        if (left < right) // 否则元素与自身交换没有意义
            swap(nums[left], nums[right]);
    }
    swap(nums[right], nums[low]); // right 指向最后一个 小于等于 pivot 的元素
    return right;
}

/**
 * 快速排序
 * @param nums 被排序数组
 * @param low 首个元素索引
 * @param high 末尾元素索引
 */
void quickSort(vector<int> &nums, int low, int high) {
    if (low < high) { // 不要忽略该判断，归并排序中也有类似判断
        int par = partition(nums, low, high);
        quickSort(nums, low, par - 1);
        quickSort(nums, par + 1, high);
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    int n;
    cin >> n;

    vector<int> nums(n);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    quickSort(nums, 0, n - 1); // 后两个参数是：首个元素的索引和末尾元素的索引

    for (int i = 0; i < n; i++) {
        cout << nums[i] << " ";
    }
    return 0;
}
```

#### 堆排

```c++
const int maxn = 100;
// heap 为堆，heap[1] 为根结点，n 为元素个数
int heap[maxn], n = 10;


// 对 heap 数组在 [low, high] 范围进行向下调整
// 其中 low 为欲调整结点的数组下标，high 一般为堆的最后一个元素的数组下标
// 时间复杂度O(log n)
void downAdjust(int low, int high) {
    int i = low, j = i * 2; // i 为欲调整结点，j  为其左孩子
    while (j <= high) { // 存在孩子结点
        // 如果右孩子存在，且右孩子的值大于左孩子
        if(j + 1 <= high && heap[j + 1] > heap[j])
            j++; // 让 j 存储右孩子下标

        // 如果孩子中的最大权值比欲调整结点i大
        if(heap[j] > heap[i]) {
            swap(heap[j], heap[i]); // 交换最大权值的孩子与欲调整结点i
            i = j; // 保持 i 为欲调整结点，j 为 i 的左孩子
            j = i * 2;
        } else break; // 孩子的权值均比欲调整结点i小，调整结束
    }
}

// 建堆：时间复杂度O(n)，证明可参考《算法导论》
void createHeap() {
    for (int i = n / 2; i >= 1; i--) {
        downAdjust(i, n);
    }
}

// 删除堆顶元素：每次删除堆中元素只能删除当前堆顶
// 时间复杂度O(log n)
void deleteTop() {
    heap[1] = heap[n--]; // 用最后一个元素覆盖堆顶元素，并让元素个数减1
    downAdjust(1, n); // 向下调整堆顶元素
}

// 向上调整：用于添加元素到堆时，元素添加到最后一个结点后进行调整
// 时间复杂度O(log n)
// 对 heap 数组在 [low, high] 范围进行向上调整
// 其中 low 一般设置为1，high 表示欲调整结点的数组下标
void upAdjust(int low, int high) {
    int i = high, j = i / 2; // i 为欲调整结点，j 为其父亲
    while (j >= low) { // 父亲在 [low, high] 范围内
        if(heap[j] < heap[i]) {
            swap(heap[j], heap[i]); // 交换父亲和欲调整结点
            i = j; // 保持 i 为欲调整结点，j 为 i 的父亲
            j = i / 2;
        } else break; // 父亲权值比欲调整结点i的权值大，调整结束
    }
}

// 添加元素x
void insert(int x) {
    heap[++n] = x; // 让元素个数加1，然后将数组末位赋值为x
    upAdjust(1, n); // 向上调整新加入的结点
}

// 堆排序
void heapSort() {
    createHeap(); // 建堆
    for (int i = n; i > 1; i--) { // 倒着枚举，直至堆中只有一个元素
        swap(heap[i], heap[1]); // 交换 heap[i] 与堆顶
        downAdjust(1, i - 1); // 调整堆顶
    }
}
```

#### 拓扑排序

```c++
// LeetCode_210 https://leetcode.com/problems/course-schedule-ii/

class Solution {
public:
    vector<int> findOrder(int numCourses, vector<pair<int, int>>& prerequisites) {
        vector<vector<int>> graph(numCourses, vector<int>());
        vector<int> inDegree(numCourses); // 入度为 0 时顶点可以删除
        vector<int> res;
        queue<int> q; // 存储入度为 0 的顶点
        int cnt = 0; // 放入 res 的元素个数

        for(auto &p : prerequisites) {
            graph[p.second].push_back(p.first);
            inDegree[p.first]++;
        }

        for(int i = 0; i < numCourses; i++)
            if(inDegree[i] == 0)
                q.push(i);

        while(!q.empty()){
            int cur = q.front();
            q.pop();

            res.push_back(cur);
            cnt++;

            for(auto j : graph[cur]){
                inDegree[j]--;
                if(inDegree[j] == 0)
                    q.push(j);
            }
        }

        if(cnt < numCourses) {
            return vector<int>();
        }
        return res;
    }
};
```

### 查找

#### 二分查找

```c++
// 二分搜索、lower_bound、upper_bound 的实现

// lower_bound.cpp

// a为递增序列，x为欲查询的数，函数返回第一个大于等于x的元素的位置
// n是a的元素个数，查询区间是[left, right] = [0, n]
int lower_bound(int a[], int left, int right, int x) {
    int mid; // mid为left和right的中点
    while (left < right) { // left==right 意味着找到唯一位置
        mid = left + (right - left) / 2;
        if (a[mid] >= x)
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}

// upper_bound.cpp

// a为递增序列，x为欲查询的数，函数返回第一个大于x的元素的位置
// n是a的元素个数，查询区间是[left, right] = [0, n]
int upper_bound(int a[], int left, int right, int x) {
    int mid; // mid为left和right的中点
    while (left < right) { // left==right 意味着找到唯一位置
        mid = left + (right - left) / 2;
        if (a[mid] > x)
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}

// 二分搜索.cpp

// a为递增序列
// n是a的元素个数，查询区间是[left, right]，传入初值是[0, n - 1]
int binarySearch(int a[], int left, int right, int x) {
    int mid; // mid为left和right的中点
    while (left <= right) { // 如果大于，无法形成闭区间
        mid = left + (right - left) / 2; // 相对地避免溢出
        if (a[mid] == x) return mid; // 找到x，返回下标
        else if (a[mid] > x)
            right = mid - 1; // 往左子区间查找
        else
            left = mid + 1; // 往右子区间查找
    }
    return -1; // 查找失败
}
```

#### 并查集

```c++
// 并查集模板：初始化、查找、合并、路径压缩


void init() {
    for(int i = 0; i < n; i++) {
       father[i] = i; // 令father[i]为-1也可
    }
}

// 递推查找
int findFather(int x) {
    while(x != father[x]) {
        x = father[x];
    }
    return x;
}

// 递归查找
int findFather(int x) {
    if (x == father[x]) return x; // 如果找到根结点，则返回根结点编号x
    else return findFather(father[x]); // 否则，递归判断x的父亲结点是否是根结点
}

// 合并
void Union(int a, int b) {
    int fa = findFather(a);
    int fb = findFather(b);
    if(fa != fb) {
        father[fa] = fb;
    }
}

// 路径压缩 迭代实现
int findFather(int x) {
    // 由于x在下面的while中会变成根结点，因此先把原x保存
    int cur = x;
    while (x != father[x])
        x = father[x];

    // x此时为根结点，下面将路径上的所有结点的father都改成根结点
    while(cur != father[cur]) {
        int z = cur; // 因为cur要被father[cur]覆盖，所以先保存
        cur = father[cur]; // 回溯父结点
        father[z] = x; // 将原先的结点的父亲改为根结点
    }
    return x; // 返回根结点
}

// 路径压缩 递归实现
int findFather(int v) {
    if(v == father[v]) return v; // 找到根结点
    father[v] = findFather(father[v]);
    return father[v];
}
```

### 位运算

-   判断一个数字x二进制下第i位是不是等于1

    -   `if(((1<<(i−1))&x)>0)`

    >   将1左移i-1位，相当于制造了一个只有第i位上是1，其他位上都是0的二进制数。然后与x做与运算，如果结果>0，说明x第i位上是1，反之则是0。

-   将一个数字x二进制下第i位更改成1。

    -   `x=x|(1<<(i−1))`

-   把一个数字二进制下最靠右的第一个1去掉

    -   `x=x&(x−1)`

-   判断n是不是2的次幂

    -   (n > 0 && ((n & (n - 1)) == 0)

    >   因为2的次幂 二进制下只有一个1 所以 减一以后 最高位变为0 后面的都变为1， &之后 就变为0了 所以本题中 i的1的个数 等于他去掉最低位的1之后的i&(i-1)的1的个数再+1

### 分数表示

```c++
// 分数的表示
// 常用假分数的形式表示
// 设分母为非负数；如果分数为负，令分子为负即可
// 分子和分母没有除了1以外的公约数，即总是约分
struct Fraction { // 分数
    int up, down; // 分子，分母
}

// 分数化简
// 如果分母为负数，分子、分母都取相反数
// 如果分子为0，令分母为1
// 约分：求出分子绝对值与分母绝对值的最大公约数d，令分子分母同时除以d
        Fraction reduction(Fraction result){
if(result.down < 0){
result.down = -result.down;
result.up = -result.up;
}
if(result.up == 0)
result.down = 1;
else{
int d = GCD(abs(result.up), abs(result.down));
result.up /= d;
result.down /= d;
}
return result;
}

// 分数的加法

Fraction add(Fraction f1, Fraction f2){
    Fraction result;
    result.up = f1.up * f2.down + f2.up * f1.down;
    result.down = f1.down * f2.down;
    return reduction(result);
}

// 分数的减法

Fraction minu(Fraction f){
    f.up = -f.up;
    return reduction(f);
}

Fraction minu(Fraction f1, Fraction f2){
    return add(f1, minu(f2));
}

// 分数的乘法

Fraction multi(Fraction f1, Fraction f2){
    Fraction result;
    result.up = f1.up * f2.up;
    result.down = f1.down * f2.down;
    return reduction(result);
}

// 分数的除法

Fraction divide(Fraction f1, Fraction f2){
    Fraction result;
    result.up = f1.up * f2.down;
    result.down = f1.down * f2.up;
    return reduction(result);
}

// 输出分数前，进行化简。
// 如果分母为1，则该分数为整数，直接输出分子。
// 如果分子绝对值大于分母，则该分数为假分数，应以带分数的形式输出。
// 若以上均不满足，则分数是真分数，按原样输出即可。
void showResult(Fraction r) {
    r = reduction(r);
    if(r.down == 1) printf("%lld", r.up); // 整数
    else if(abs(r.up) > r.down) { // 假分数
        printf("%d %d/%d", r.up / r.down, abs(r.up) % r.down, r.down);
    }
    else {
        printf("%d/%d", r.up, r.down);
    }
}
```

### 字符串

#### 回文串

```c++
// 回文串判断.cpp

#include <iostream>
#include <cstring>

const int maxn = 256;

using namespace std;

bool judge(char str[]){
    int len = strlen(str); // 字符串长度
    for (int i = 0; i < len / 2; i++) { // 枚举字符串前一半
        if(str[i] != str[len - i - 1])
            return false;
    }
    return true;
}

int main(){
    char str[maxn];
    while(cin >> str) { // 输入字符串
        bool flag = judge(str); // 判断字符串str是否是回文串
        if(flag)
            cout << "YES" << endl;
        else
            cout << "NO" << endl;
    }
    return 0;
}
```

### 树算法

#### 哈夫曼树

给定n个权值作为n个叶子结点，构造一棵二叉树，若该树的带权路径长度达到最小，称这样的二叉树为最优二叉树，也称为哈夫曼树(Huffman Tree)。哈夫曼树是带权路径长度最短的树，权值较大的结点离根较近。

路径长度：两点路径上的分枝数目称作路径长度。

树的路径长度：从树根到每个结点的路径长度之和。

结点的带权路径长度：在一棵树中，假设其结点上附带有一个权值，通常把该结点的路径长度与该结点上的权值

```c++
/*
题目大意：FJ需要修补牧场的围栏，他需要 N 块长度为 Li 的木头（N planks of woods）。开始时，FJ只有一块无限长的木板，因此他需要把无限长的木板锯成 N 块长度 为 Li 的木板，Farmer Don提供FJ锯子，但必须要收费的，收费的标准是对应每次据出木块的长度，比如说测试数据中 5 8 8

他希望将长度为21的木板切成长度为8,5和8的木板
8 + 5 + 8 = 21。第一次切割将花费21美元，并且应该用于将板切割成13和8片。第二次切割将花费13，并且应该用于将13切割成8和5.这将花费21 + 13 = 34 。如果21被切割成16和5，则第二次切割将花费16次，总共37（大于34）。

题目可以转化为Huffman树构造问题 / 优先队列：
通过每次选取两块长度最短的木板，合并，最终必定可以合并出长度为 Sum（Li）的木板，并且可以保证总的耗费最少
 * */
#include <cstdio>
#include <queue>

using namespace std;

priority_queue<long long, vector<long long>, greater<long long> > pq; // 小顶堆

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    int n;
    long long temp, x, y, ans = 0;

    // 输入到优先级队列
    cin >> n;
    for(int i = 0; i < n; i++) {
        cin >> temp;
        pq.push(temp);
    }

    while(pq.size() > 1) { // 只要堆中至少有两个元素，就继续执行合并
        x = pq.top();
        pq.pop();
        y = pq.top();
        pq.pop();

        pq.push(x + y); // 取出栈顶
        ans += (x + y);        
    }

    cout << ans << '\n';
    return 0;
}
```

### 图算法

#### 最短路径

```c++
// Dijkstra.邻接矩阵.cpp

#include <iostream>
#include <algorithm>

using namespace std;
#define MAXV 1010
#define INF 0x3f3f3f3f

int n, m, s; // 顶点个数，边数，起点编号
int G[MAXV][MAXV]; // MAXV为最大顶点数
int d[MAXV]; // 起点到达各点的最短路径长度
bool vis[MAXV] = {false}; // 表示是否访问的数组

void Dijkstra(int s) { // s为起点
    fill(d, d + MAXV, INF); // 相当于C语言的memset
    d[s] = 0; // 起点到达自身的距离为0，不要把起点标记为已访问
    for (int i = 0; i < n; i++) { // 循环次数为顶点的数量
        int u = -1, MIN = INF; // 找到u使得d[u]最小，MIN存放该最小的d[u]
        for (int j = 0; j < n; j++) {
            if (vis[j] == false && d[j] < MIN) {
                u = j;
                MIN = d[j];
            }
        }
        // 找不到小于INF的d[i]，说明剩下的顶点和起点s不连通
        if (u == -1) return;
        vis[u] = true; // 标记u为已访问
        for (int v = 0; v < n; v++) {
            // 如果v未访问 且 u能到达v 且 以u为中介点可以使d[v]更优
            if (vis[v] == false && G[u][v] != INF && d[u] + G[u][v] < d[v]) {
                d[v] = d[u] + G[u][v];
            }
        }
    }
}

int main() {
    int u, v, w;
    cin >> n >> m >> s; // 顶点个数，边数，起点编号
    fill(G[0], G[0] + MAXV * MAXV, INF); // 初始化图G
    for (int i = 0; i < m; i++) {
        cin >> u >> v >> w;
        G[u][v] = w;
    }
    Dijkstra(s); // Dijkstra 算法入口
    for (int i = 0; i < n; i++) {
        cout << d[i] << " "; // 输出起点到所有顶点的最短距离
    }
    return 0;
}

// Floyd.cpp
int dis[MAXV][MAXV]; // dis[i][j] 表示顶点i到顶点j的距离

void Floyd() {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dis[k][i] != INF && dis[k][j] != INF && dis[i][k] + dis[k][j] < dis[i][j]) {
                    dis[i][j] = dis[i][k] + dis[k][j];
                }
            }
        }
    }
}

// Bellman-Ford.cpp

struct Node {
    int v, dis; // v为邻接边的目标顶点，dis 为邻接边的边权
}

        vector<Node>
Adj[MAXV]; // 图G的邻接表
int n; // n为顶点数，MAXV 为最大顶点数
int d[MAXV]; // 起点到达各点的最短路径长度

bool Bellman(int s) { // s为源点
    fill(d, d + MAXV, INF); // fill函数将整个d数组赋为INF，注意慎用INF
    d[s] = 0; // 起点s到达自身的距离为0

    for (int i = 0; i < n - 1; i++) { // 执行 n-1 轮操作，n 为顶点数
        for (int u = 0; u < n; u++) { // 每轮操作都遍历所有边
            for (int j = 0; j < Adj[u].size(); j++) {
                int v = Adj[u][j].v; // 邻接边的顶点
                int dis = Adj[u][j].dis; // 邻接边的边权
                if (d[u] + dis < d[v]) {
                    d[v] = d[u] + dis;
                }
            }
        }
    }

    // 以下为判断负环的代码
    for (int u = 0; u < n; u++) { // 对每条边进行判断
        for (int j = 0; j < Adj[u].size(); j++) {
            int v = Adj[u][j].v; // 邻接边的顶点
            int dis = Adj[u][j].dis; // 邻接边的边权
            if (d[u] + dis < d[v]) // 如果仍可以被松弛
                return false; // 说明图中有从源点可达的负环
        }
    }
    return true; // 数组d的所有值都已经达到最优
}

// SPFA.cpp
vector <Node> Adj[MAXV]; // 图G的邻接表
int n, d[MAXV], num[MAXV]; // num数组记录顶点的入队次数
bool inq[MAXV]; // 顶点是否在队列中

bool SPFA(int s) { // s为源点
    // 初始化
    metset(inq, false, sizeof(inq));
    memset(num, 0, sizeof(num));
    fill(d, d + MAXV, INF);

    // 源点入队部分
    queue<int> Q;
    Q.push(s); // 源点入队
    inq[s] = true; // 源点已入队
    num[s]++; // 源点入队次数加1
    d[s] = 0; // 源点的距离为 0

    while (!Q.empty()) {
        int u = Q.front(); // 队首顶点编号为 u
        Q.pop(); // 出队
        inq[u] = false; // 设置 u 为不在队列中
        // 遍历 u 的所有邻接边 v
        for (int j = 0; j < Adj[u].size(); j++) {
            int v = Adj[u][j].v;
            int dis = Adj[u][j].dis;
            // 松弛操作
            if (d[u] + dis < d[v]) {
                d[v] = d[u] + dis;
                if (!inq[v]) { // 如果 v 不在队列中
                    Q.push(v); // v 入队
                    inq[v] = true; // 设置 v 为在队列中
                    num[v]++;
                    if (num[v] >= n) return false; // 有可达负环
                }
            }
        }
    }
    return true; // 无可达负环
}
```

#### 最小生成树

Prim 算法和 Kruskal 算法两者都是贪心算法，但两者都能获得最小生成树的精确最优解。前者是寻找顶点的过程，更适合稠密图；后者是寻找边的过程，更适合稀疏图。

##### PRIM 算法[¶](https://looperxx.github.io/My_wiki/Algorithm Details/#prim)

Prim 算法的关键两步是寻找顶点与更新最短距离。 该算法设定集合S，尝试不断向S中添加顶点来生成最小生成树。当找到S之外距离S集合最近的顶点u后，更新u的所有邻接顶点到集合S的最短距离，并不断循环添加顶点。直到找不到这样的顶点u，即剩余顶点不与S连通，或所有顶点都已被加入到最小生成树。

邻接矩阵的实现：

```c++
int n, G[maxv][maxv]; // n为顶点数量，maxv 为最大顶点数
int d[maxv]; // 顶点与集合S的最短距离
bool vis[maxv]; // 标记数组

int prim() { // 默认 0 号为初始点，函数返回最小生成树的边权之和
    fill(d, d + maxv, INF);
    d[0] = 0;
    int ans = 0; // 存放最小生成树的边权之和
    for(int i = 0; i < n; i++) {
        int u = -1, MIN = INF; 
        for(int j = 0; j < n; j++) { // 找到未访问的顶点中距离当前生成的集合最近的顶点
            if(!vis[j] && d[j] < MIN) {
                u = j;
                MIN = d[j];
            } 
        }
        if(u == -1) return -1; // 找不到顶点，即剩下的顶点与集合S不连通
        vis[u] = true;
        ans += d[u]; // 最小生成树的边权累加
        // 更新剩余顶点到集合S的距离
        for(int v = 0; v < n; v++) {
            if(!vis[v] && G[u][v] != INF && G[u][v] < d[v]) {
            d[v] = G[u][v];
            }
        }
    }
    return ans;
}
```

邻接表的实现：

```c++
class Edge {
    int to, dis; // 目标顶点和边权
}
vector<Edge> nei[maxv]; // 邻接表，nei[u]存放顶点u出发的边

int n; // 顶点数
int d[maxv]; // 顶点到集合S的最短距离
bool vis[maxv]; // 标记数组

int prim() { // 默认0号为初始点，函数返回最小生成树的边权之和
    fill(d, d + maxv, INF);
    d[0] = 0;
    int ans = 0;
    for(int i = 0; i < n; i++) {
        int u = -1, MIN = INF;
        for(int j = 0; j < n; j++) {
            if(!vis[j] && d[j] < MIN) {
                u = j;
                MIN = d[j];
            }
        }
    }

    if(u == -1) return -1;
    vis[u] = true;
    ans += d[u];
    // 只有下面这个循环与邻接矩阵的实现不同
    for(int j = 0; j < nei[u].size(); j++) {
        int v = nei[u][j].to;
        int dis = nei[u][j].dis;
        if(!vis[v] && dis + d[u] < d[v]) {
            d[v] = dis + d[u];
        }
    }
    return ans;
}
```

##### KRUSKAL 算法

Kruskal 算法使用并查集。先将图中所有边按照边权从小到大进行排序，然后开始遍历每条边，通过并查集将判断并将边添加到最小生成树的集合。

```c++
struct Edge{
    int u, v; // 边的两个端点编号
    int cost; // 边权
} edges[maxe]; // 最多有 maxe 条边

bool cmp(Edge a, Edge b) {
    return a.cost < b.cost;
}

// 并查集部分
int F[maxv]; // 并查集数组
int find(int x) { // 并查集查询函数
    int a = x;
    while (x != F[x]) {
        x = F[x];
    }
    // 路径压缩
    while(a != F[a]) {
        int z = a;
        a = F[a];
        F[z] = x;
    }
    return x;
}

/**
 * Kruskal 算法
 * @param n 顶点个数
 * @param m 图的边数
 * @return 所求边权只和；若无法连通返回-1
 */
int Kruskal(int n, int m) {
    // ans 为所求边权之和，numEdge 为当前生成树的边数
    int ans = 0, numEdge = 0;
    for(int i = 0; i < n; i++) { // 顶点范围是[0...n-1]
        F[i] = i; // 并查集初始化
    }
    sort(edges, edges + m, cmp); // 所有边按照边权排序
    for(int i = 0; i < m; i++) { // 枚举所有边
        int faU = find(edge[i].u); // 查询测试边两个端点所在集合的根结点
        int faV = find(edge[i].v);
        if(faU != faV) { // 如果不在一个集合中，该边尚未加入到最小生成树中
            F[faU] = faV; // 合并集合，将测试边加入到最小生成树中
            ans += edges[i].cost; 
            numEdge++; // 当前生成树的边数加1
            if(numEdge == n - 1) // 所有边都已添加
                break;
        }
    }
    if(numEdge != n - 1)
        return -1; // 无法连通时返回 -1
    else return ans; // 返回最小生成树的边权之和
}

int main() {
    int n, m; // 顶点数量，边数量
    cin >> n >> m;
    for(int i = 0; i < m; i++)
        cin >> edges[i].u >> edges[i].v >> edges[i].cost; // 两个端点编号，边权
    int ans = Kruskal(n, m);
    cout << ans << endl;
    return 0;
}
```

### 其他

#### 斐波那契与卡塔兰数

#### 欧几里得定理

>   <https://www.cnblogs.com/fisherss/p/9985135.html>

#### 矩阵连乘

```c++
#include <iostream>
#define maxn 100
#define inf 0x3f3f3f3f

int p[maxn];
int dp[maxn][maxn]; // 初始对角线为0
int s[maxn][maxn]; // 切割矩阵链，存储前半段矩阵链的最后一个矩阵的索引
int main() {
    int n = 6; // 连乘的矩阵数量
    p[0] = 30;
    p[1] = 35;
    p[2] = 15;
    p[3] = 5;
    p[4] = 10;
    p[5] = 20;
    p[6] = 25;

    for(int i = 1; i <= n; i++) {
        for(int j = 1; j <= n; j++) {
            if(j > i) dp[i][j] = inf; // 上对角线初始化为 inf
        }
    }

    for(int r = 2; r <= n; r++) { // 每次循环的矩阵链的长度
        for(int i = 1; i <= n - r + 1; i++) { // i 是矩阵链开始
            int j = i + r - 1; // 当前矩阵链是矩阵i到矩阵j
            for(int k = i; k < j; k++) { // k 是矩阵链切割位置
                int t = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j];
                if(t < dp[i][j]) {
                    dp[i][j] = t;
                    s[i][j] = k;
                }
            }
        }
    }


    for(int i = 1; i <= n; i++) {
        for(int j = 1; j <= n; j++) {
            cout << dp[i][j] << "\t";
        }
        cout << '\n';
    }

    cout << '\n';

    for(int i = 1; i <= n; i++) {
        for(int j = 1; j <= n; j++) {
            cout << s[i][j] << "\t";
        }
        cout << '\n';
    }
}
```

#### 快速乘法

```c++
typedef long long LL;

LL mul(LL a, LL b, LL m) {
    LL ans = 0; 
    while (b) {
        if (b & 1)
            ans = (ans + a) % m;
        a = (a + a) % m;
        b >>= 1;
    }
    return ans;
}

// 标量快速幂 迭代

typedef long long LL;

LL binaryPow(LL a, LL b, LL m) {
    LL ans = 1;
    while (b) {
        if (b & 1)
            ans = ans * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return ans;
}

// 标量快速幂.递归

// 求 a^b % m，递归写法

LL binaryPow(LL a, LL b, LL m) {
    if (!b) return 1; // 如果b为0，则a为1
    if (b & 1)
        return a * binaryPow(a, b - 1, m) % m;
    LL mul = binaryPow(a, b / 2, m);
    return mul * mul % m;
}

// 矩阵快速幂.cpp

typedef long long LL;
vector <LL> vec; // 长整数向量
vector <vec> mat; // 长整数矩阵

mat mul(const mat &A, const mat &B) { // 矩阵乘法 A multiply B
    mat C(A.size(), vec(B[0].size())); // 大小为 A 的行数与 B 的列数
    for (int i = 0; i < A.size(); i++) {
        for (int k = 0; k < B.size(); k++) {
            for (int j = 0; j < B[0].size(); j++) {
                Ci = (Ci + (Ai * Bk) % MOD) % MOD;
                // 如果必要，此处的乘法以快速乘法代替
            }
        }
    }
    return C;
}

mat pow(mat A, LL n) { // 计算矩阵 A^n
    // 矩阵乘法的零元是单位矩阵
    mat An(A.size(), vec(A.size()));
    for (int i = 0; i < A.size(); i++)
        Ani = 1;
    while (n > 0) {
        if (n & 1)
        An = mul(A, An);
        A = mul(A, A);
        n >>= 1;
    }
    return An;
}
```

#### 素数

```c++
// 素数筛判断 x到y的素数个数 （包括x和y）

#include <cmath>
#include <iostream>

using namespace std;

const int MAXY = 1e5 + 10;
bool comp[MAXY]; // 合数为 true

int main() {
    int x, y;
    int ans = 0;
    cin >> x >> y;
    // 注意，此题没有说明 x 和 y 的大小关系
    if (x > y) {
        swap(x, y);
    }
    // 埃式筛法
    int m = sqrt(y + 0.5);
    for (int i = 2; i <= m; i++) {
        if (!comp[i]) { // 质数
            for (int j = i * i; j <= y; j += i)
                comp[j] = true;
        }
    }

    // 统计素数个数
    for (int i = x; i <= y; i++) {
        if (i != 1 && !comp[i]) // 注意 1 不是质数
            ans++;
    }
    cout << ans << endl;
    return 0;
}

// 求解质因数个数 不重复

int getPrimeNum(int val){
    int old = val;
    int res = 0;
    for(int i = 2; i <= sqrt(val + 0.5); i++){
        if(val % i == 0){
            res++;
            while(val % i == 0)
                val /= i;

        }
    }
    if(val > 1 && val != old) res++; // 不包括自身时 需要加上 val != old
    return res;
}
```

#### 逆序对

逆序对：设 A 为一个有 n 个数字的有序集 (n>1)，其中所有数字各不相同。如果存在正整数 i, j 使得 1 ≤ i < j ≤ n 而且 A[i] > A[j]，则 A[i], A[j]> 这个有序对称为 A 的一个逆序对，也称作逆序数。

##### 归并排序

```
归并排序：（即使序列存在相同元素，该算法也适用，且代码不用修改） 
归并排序是将数列a[l,h]分成两半a[l,mid]和a[mid+1,h]分别进行归并排序，然后再将这两半合并起来。
在合并的过程中（设l<=i<=mid ,mid+1<=j<=h), 当a[i]<=a[j]时，并不产生逆序数；
当a[i]>a[j]时，在前半部分中比a[i]大的数都比a[j]大，将a[j]放在a[i]前面的话，逆序数要加上mid+1-i。
因此，可以在归并排序中的合并过程中计算逆序数.
```

```c++
#include<bits/stdc++.h>
#define LL long long
using namespace std;
const int MAXN = 50000+5;
const int MAXM = 1e5;
int a[MAXN],temp[MAXN];
LL ans=0;
int n;
void merge(int le,int mid,int ri){//使每两部分【都已经分别有序】，合并为一个有序集合 
    int i,j,k;
    // le到mid 是一个有序部分，mid+1到ri是一个有序部分 ，合并就行了  
    i=le;j=mid+1;k=le;
    for(;i<=mid&&j<=ri;){
        if(a[i]>a[j]){
            temp[k++]=a[j++];
            ans+=mid-i+1;
        }else  temp[k++]=a[i++];
    }
    while(i<=mid) temp[k++]=a[i++];
    while(j<=ri) temp[k++]=a[j++];
    for(i=le;i<=ri;i++) a[i]=temp[i];
}
void merge_sort(int le,int ri){//不断的分为一半，来使各个部分非递减有序 
    if(le<ri){
        int  mid=(le+ri)>>1;
        merge_sort(le,mid);
        merge_sort(mid+1,ri);
        merge(le,mid,ri);
    }
}
int main(){
    while(~scanf("%d",&n)){
        for(int i=0;i<n;i++) scanf("%d",&a[i]);
        ans=0;
        merge_sort(0,n-1);
        printf("%lld\n",ans);
    }
    return 0;
}
```

##### 线段树

##### 树状数组

<https://blog.csdn.net/qq_37383726/article/details/76459527>

## 动态规划

### 背包问题

#### 背包板子

```c++
#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdio>
#define maxn 1000;
using namespace std;

int cash, n;
int dp[maxn];
int c[maxn], w[maxn], num[maxn];

void ZeroOne_Pack(int cost, int weight) {
  for(int i = cash;i >= cost;i--) dp[i] = max(dp[i], dp[i-cost]+weight);
}

void Complete_Pack(int cost, int weight) {
  for(int i = cost;i <= cash;i++) dp[i] = max(dp[i], dp[i-cost]+weight);
}

int Multi_Pack() {
  memset(dp, 0, sizeof(dp));
  for(int i = 1;i <= n;i++) {
    if(num[i]*c[i] > cash) { //物品全装进去超重，说明物品无限，直接套完全背包
      Complete_Pack(c[i], w[i]);
    }
    else {
      int k = 1;
      while(k < num[i]) { //利用二进制思想，把物品分成1，2，4，8个，依次01背包，能凑出任何数值，降复杂度
        ZeroOne_Pack(k*c[i], k*w[i]);
        num[i] -= k;
        k <<= 1;
      }
      ZeroOne_Pack(num[i]*c[i], num[i]*w[i]);
    }
  }
  return dp[cash];
}
```

#### Poj1276[多重背包]

>   [关于01背包一维时的逆序问题](https://blog.csdn.net/xiajiawei0206/article/details/19933781)

-   二进制转化
    -   转换成01背包，例如一件物品可以拿26件，每一件的权值是w，26可以写成（1+2+4+8）+11，所以就把这种物品分解成权重为w，2w，4w，8w，11w的五种物品，这五种物品组合，一定能组成小于等于26的任意一个数，这样就把有26件的一种物品换成了五种各有一件的物品，用01背包就能解决了
    -   慢 内存占用小
    -   多开空间

```c++
#include <iostream>
#include <cstring>

using namespace std;

int cash, N; // N是钞票种数
int n[13], D[13]; // n是钞票数，D是钞票面额；
int dp[100010];

int main() {
    while (scanf("%d%d", &cash, &N) != EOF) { // cash 是金额上限，本题的重量和价值合一，金额上限可理解为背包重量上限
        // 输入与初始化
        for (int i = 0; i < N; i++)
            scanf("%d%d", &n[i], &D[i]);
        memset(dp, 0, sizeof(dp));
        // 多重背包
        for (int i = 0; i < N; i++) {  // 第i种钞票
            if (n[i] * D[i] > cash) { // 钞票充足，套用完全背包
                for (int j = D[i]; j <= cash; j++) { // j是重量
                    dp[j] = max(dp[j], dp[j - D[i]] + D[i]);
                }
            } else { // if n[i] * D[i] <= cash
                int num = n[i]; // 剩余钞票数
                for (int k = 1; num > 0; k <<= 1) {  // 钞票数；使用二进制优化，k为2的幂，依次0-1背包，可凑出任意数值
                    int mul = min(k, num);
                    for (int j = cash; j >= mul * D[i]; j--) { // j是重量
                        dp[j] = max(dp[j], dp[j - mul * D[i]] + mul * D[i]);
                    }
                    num -= mul;
                }
            }
        }
        // 输出
        printf("%d\n", dp[cash]);
    }
    return 0;
}
```

维护一个数组num，$num[i][j]num[i][j]$ 表示在往背包里试着装第i件物品时，背包容量使用了j时装了多少件i物品。cost[i]为物品i的体积，$num[i][j]=num[j−cost[i]]+1num[i][j]=num[j−cost[i]]+1$ ;可以看到$num[i][j]num[i][j]$和$num[t|t!=i][j|j=1,2,....v]num[t|t!=i][j|j=1,2,....v]$没有任何关系，所以可以省掉第一维，对于每一种物品，清空num，只计算num[j]就行了

-   快 内存占用大

```c++
#include<iostream>
#include<cstdio>
#include<cstring>
#include<string>
#include<algorithm>
#include<map>
#include<queue>
#include<climits>
#include<list>
#include<stack>
#define mem(a) memset(a,0,sizeof(a))

using  namespace std;

int wi[20];
int mi[20];
int dp[100050];
int num[100050];

int main()
{
    int ans_v;
    while(scanf("%d",&ans_v)!=EOF)
    {
        int n;
        mem(dp);
        mem(num);
        scanf("%d",&n);
        int i;
        for(i=0;i<n;++i)
        {
            scanf("%d%d",&mi[i],&wi[i]);
        }
        for(i=0;i<n;++i)
        {
            int j;
            mem(num);
            for(j=wi[i];j<=ans_v;++j)
            {
                if(dp[j-wi[i]]+wi[i]>dp[j]&&num[j-wi[i]]<mi[i])
                {
                    dp[j]=dp[j-wi[i]]+wi[i];
                    num[j]=num[j-wi[i]]+1;
                }
            }
        }
        printf("%d\n",dp[ans_v]);
    }
}
```

#### Poj2184[01背包变形]

-   使得TS和TF的值的和最大，同时TS和TF都不能为0

```c++
#include <iostream>
#include <climits>

#define MAXN 105
using namespace std;
int N;
const int M = 100005; // TS或TF最大为100000
int S[MAXN], F[MAXN];
int dp[2 * M]; // dp[i]表示S值为i时F的最大值；坐标平移：负数在前，正数在后
// 将S看做代价，F看做价值，0-1背包求解

int main() {
    while (scanf("%d", &N) != EOF) {
        for (int i = 0; i < N; i++)
            scanf("%d%d", &S[i], &F[i]);
        for (int i = 0; i < 2 * M; i++)
            dp[i] = INT_MIN;
        dp[100000] = 0; // 0前有十万个负数，0后有十万个正数

        for (int i = 0; i < N; i++) {
            if (S[i] >= 0) {
                for (int j = 2 * M - 1; j >= S[i]; j--) {
                    if (dp[j - S[i]] > INT_MIN) // 背包重量为 j - S[i] 的情况存在
                        dp[j] = max(dp[j], dp[j - S[i]] + F[i]);
                }
            } else { // if S[i] < 0，如果放入背包，背包变轻，逆序遍历
                for (int j = S[i]; j - S[i] < 2 * M; j++) {
                    if (dp[j - S[i]] > INT_MIN)
                        dp[j] = max(dp[j], dp[j - S[i]] + F[i]);
                }
            }
        }

        int ans = INT_MIN;
        for (int i = 100000; i < 2 * M; i++) {
            if (dp[i] >= 0) //TF值大于0 显然这里的TS=i-100000>0 
                ans = max(ans, dp[i] + i - 100000); // dp[i] + i 即 TS + TF，然后恢复坐标平移
        }
        printf("%d\n", ans);
    }
    return 0;
}
```

### 状压DP

>   <https://www.cnblogs.com/Tony-Double-Sky/p/9283254.html>

状压就是将问题可能遇到的每一个状态用一个唯一的二进制数表示

状压位运算

-   判断一个数字x二进制下第i位是不是等于1。
    -   方法：if(((1<<(i−1))&x)>0)
    -   将1左移i-1位，相当于制造了一个只有第i位上是1，其他位上都是0的二进制数。然后与x做与运算，如果结果>0，说明x第i位上是1，反之则是0。
-   将一个数字x二进制下第i位更改成1。
    -   方法：x=x|(1<<(i−1))
    -   证明方法与1类似，此处不再重复证明。
-   把一个数字二进制下最靠右的第一个1去掉。
    -   方法：x=x&(x−1)

### 线段树

```c++
// https://leetcode.com/problems/range-sum-query-mutable/ 线段树保存可变区间和

// 数组法
class NumArray {
public:
    NumArray(vector<int> nums) {
        n = nums.size();
        tree.resize(n * 2);
        buildTree(nums);
    }

    void buildTree(vector<int>& nums) {
        for (int i = n; i < n * 2; ++i) {
            tree[i] = nums[i - n];
        }
        for (int i = n - 1; i > 0; --i) {
            tree[i] = tree[i * 2] + tree[i * 2 + 1];
        }
    }

    void update(int i, int val) {
        tree[i += n] = val;
        while (i > 0) {
            tree[i / 2] = tree[i] + tree[i ^ 1];
            i /= 2;
        }
    }

    int sumRange(int i, int j) {
        int sum = 0;
        for (i += n, j += n; i <= j; i /= 2, j /= 2) {
            if ((i & 1) == 1) sum += tree[i++];
            if ((j & 1) == 0) sum += tree[j--];
        }
        return sum;
    }

private:
    int n;
    vector<int> tree;
};

// 构造SegmentTreeNode类
/*
 * 本题实现的函数：NumArray 类的构造函数，sumRange，update
 * 本代码使用线段树实现该类，NumArray 包含一个子类 SegmentTreeNode
 */

class NumArray {
public:
    // 成员变量与子类
    class SegmentTreeNode {
    public:
        int s, e;
        SegmentTreeNode *left, *right;
        int sum;

        // 线段树构造函数
        SegmentTreeNode(int s, int e) :
                s(s), e(e), left(nullptr), right(nullptr), sum(0) {
        }

        // 线段树析构函数
        ~SegmentTreeNode() {
            if (this->left != nullptr)
                delete (this->left);
            if (this->right != nullptr)
                delete (this->right);
            delete this;
        }
    };

    SegmentTreeNode *root = nullptr;

    // 成员方法

    /*
     * 类的构造函数
     * @param nums 数组
     */
    NumArray(vector<int> nums) {
        root = buildTree(nums, 0, nums.size() - 1);
    }

    /*
     * 线段树的构造
     * @param nums 数组
     * @param s 起始索引
     * @param e 终止索引
     */
    SegmentTreeNode *buildTree(vector<int> nums, int s, int e) {
        if (s > e)
            return nullptr;
        SegmentTreeNode *ret = new SegmentTreeNode(s, e);
        if (s == e) {
            ret->sum = nums[s];
        } else {
            int mid = s + (e - s) / 2; // 中位
            ret->left = buildTree(nums, s, mid);
            ret->right = buildTree(nums, mid + 1, e);
            ret->sum = ret->left->sum + ret->right->sum;
        }
        return ret;
    }

    void update(int i, int val) {
        update(root, i, val);
    }

    void update(SegmentTreeNode *root, int pos, int val) {
        if (root->s == root->e) // 如果线段树只有一个数
            root->sum = val;
        else {
            int mid = root->s + (root->e - root->s) / 2; // 线段树中位
            if (pos <= mid) update(root->left, pos, val); // 左子树中更新，注意取等
            else update(root->right, pos, val); // 右子树中更新
            root->sum = root->left->sum + root->right->sum; // 更新各层结点sum
        }
    }

    int sumRange(int i, int j) {
        return sumRange(root, i, j);
    }

    int sumRange(SegmentTreeNode *root, int i, int j) {
        if (i > j) swap(i, j);
        if (root->e == j && root->s == i)
            return root->sum;
        else {
            int mid = root->s + (root->e - root->s) / 2; // 线段树中位
            if (mid >= j) return sumRange(root->left, i, j); // 左子树中求和
            else if (mid < i) return sumRange(root->right, i, j); // 右子树中求和
            else return sumRange(root->left, i, mid) + sumRange(root->right, mid + 1, j);
        }
    }
};


// 树状数组
class BinaryxexTree {
public:
    BinaryxexTree(vector<int>& s) {
        buildTree(s);
    }

    BinaryxexTree() {};

    void buildTree(vector<int>& s) {
        size = s.size();
        nums = vector<int>(size, 0);
        bit = vector<int>(size + 1, 0);
        for(int i = 0; i < s.size(); ++i)
            update(i, s[i]);
        nums = s;
    }

    void update(int x, int n) {
        int val = n - nums[x];
        nums[x] = n;
        for(int i = x + 1; i <= size; i += lowbit(i)) {
            bit[i] += val;
        }
    }

    int sumRange(int i, int j) {
        if(j < i || i < 0) return 0;
        if(i == j) return nums[i];
        return getSum(j) - getSum(i - 1); // i - 1 不会越出下界
    }

private:
    int lowbit(int x) {
        return x & (-x);
    }

    int getSum(int x) {
        int sum = 0;
        for(int i = x + 1; i; i -= lowbit(i)) {
            sum += bit[i];
        }
        return sum;
    }
    vector<int> bit;
    vector<int> nums;
    int size;
};

class NumArray {
public:

    NumArray(vector<int> nums) {
        tree.buildTree(nums);
    }

    void update(int i, int val) {
        tree.update(i, val);
    }

    int sumRange(int i, int j) {
        return tree.sumRange(i, j);
    }

private:
    BinaryxexTree tree;
};
```

#### 扫描线

#### 线段树+扫描线

```c++
// 蓝桥杯 2017_10 线段树+扫描线 经典题

// https://blog.csdn.net/konghhhhh/article/details/78236036

#include <stdio.h>
#include <algorithm>

using namespace std;
struct Line
{
    int x1, x2, h, f; //左右坐标，纵坐标（高度），f=1为入边，f=-1为出边
    Line() {}
    Line(int _x1, int _x2, int _h, int _f) : x1(_x1), x2(_x2), h(_h), f(_f) {}
    bool operator<(const Line &l1)
    {
        return h < l1.h;
    }
};

struct SegTree
{
    int pl, pr, cnt, len; // 左端点编号，右端点编号，被覆盖次数，两个端点之间被覆盖的长度
    SegTree *lson, *rson;
    SegTree() : cnt(0), len(0) {}
};

const int N = 10000;
int n, ans;
int X[N << 1]; //记录所有的横坐标
Line lines[N];

SegTree *buildTree(int pl, int pr)
{
    SegTree *p = new SegTree();
    p->pl = pl;
    p->pr = pr;
    if (pl == pr)
        return p;
    int mid = ((pl + pr) >> 1);
    p->lson = buildTree(pl, mid);
    p->rson = buildTree(mid + 1, pr);
    return p;
}

void updateLength(SegTree *pTree, int tl, int tr)
{
    if (pTree->cnt)
    {
        pTree->len = X[tr] - X[tl - 1]; //将区间树上的端点（序号）反入到X中求得二维坐标上的实际横坐标
    }
    else if (tl == tr)
    {
        pTree->len = 0;
    }
    else
    { //负数
        pTree->len = pTree->lson->len + pTree->rson->len;
    }
}

void update(SegTree *tree, int pl, int pr, int value) // 自底向上的更新cnt和len
{
    int tl = tree->pl;
    int tr = tree->pr;
    if (pl <= tl && pr >= tr)
    {
        tree->cnt += value;
        updateLength(tree, tl, tr);
        return;
    }
    int m = (tl + tr) >> 1;
    if (pl <= m)
        update(tree->lson, pl, pr, value);
    if (pr > m)
        update(tree->rson, pl, pr, value);
    updateLength(tree, tl, tr);
}

int main()
{
    scanf("%d", &n);
    int x1, x2, y1, y2;
    int index = 0;
    for (int i = 0; i < n; i++)
    {
        scanf("%d %d %d %d", &x1, &y1, &x2, &y2);
        X[index] = x1;
        lines[index++] = Line(x1, x2, y1, 1);
        X[index] = x2;
        lines[index++] = Line(x1, x2, y2, -1);
    }
    sort(X, X + index);
    sort(lines, lines + index);
    // 初始化线段树 先对index去重 然后得到X_end个不重复的横坐标值 所以共有X_end - 1 个离散的区间
    int X_end = unique(X, X + index) - X;
    SegTree *root = buildTree(1, X_end);

    for (int i = 0; i < index; i++)
    {
        int pl = lower_bound(X, X + X_end, lines[i].x1) - X;
        int pr = lower_bound(X, X + X_end, lines[i].x2) - X;
        update(root, pl + 1, pr, lines[i].f);
        ans += root->len * (lines[i + 1].h - lines[i].h);
    }
    printf("%d\n", ans);
    return 0;
}
```

### 树状数组

>   <https://blog.csdn.net/bestsort/article/details/80796531>

## 高精度

### 高精度进制转换

```c++
// 了解几个重要的 ASCII 码
// '0': 48
// 'A': 65
// 'a': 97

#include <iostream>
#include <cstring>

using namespace std;

const int maxn = 1000;
int t[maxn];
int d[maxn]; // 转换过程中的十进制数
char str1[maxn], str2[maxn]; // 源进制数与目标进制数
int n, m; // 源进制与目标进制

void solve() {
    int i, len, k;
    len = strlen(str1);
    for (i = len - 1; i >= 0; --i) { // t[0] 是 t 最低位，str1[0] 是 str1 的最高位
        int cur;
        if(str1[i]  <= '9')                 // str1[i] 是数字
            cur = str1[i] - '0';
        else if(str1[i] < 'a')              // str1[i] 是大写字母 
            cur = str1[i] - ('A' - 10);
        else                                // str1[i] 是小写字母 
            cur = str1[i] - ('a' - 36);
        t[len - 1 - i] = cur;
    }

    for (k = 0; len;) {
        for (i = len - 1; i >= 0; --i) { // 从 t 高位开始
            t[i] += t[i + 1] % m * n;
            t[i + 1] /= m;
        }
        d[k++] = t[0] % m;
        t[0] /= m;
        while (len > 0 && !t[len - 1]) len--; // 去掉高位的 0
    }

    str2[k] = '\0';
    for (i = 0; i < k; i++)  { // str2[0] 是 str2 的最高位，循环从低位开始
        char cur; 
        if(d[i] < 10) // 将以数字表示 
            cur = (char)(d[i] + '0');
        else if(d[i] < 36)  // 将以大写字母表示
            cur = (char)(d[i] - 10 + 'A');
        else  // 将以小写字母表示 
            cur = (char)(d[i] - 36 + 'a');
        str2[k - 1 - i] = cur;
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    int T;
    cin >> T; 
    while (T--) {
        cin >> n >> m >> str1; // 源进制，目标进制，源进制数字
        solve();
        cout << n << " " << str1 << '\n' << m << " " << str2 << '\n' << '\n';
    }
    return 0;
}
```

### 大整数

```c++
struct bign { // big number
    int d[1000]; // 存储大数，d[0]是最低位
    int len; // 记录长度

    bign() { // 构造方法
        fill_n(d, sizeof(d), 0); // 这一步不可忽略！
        // fill_n 来自 #include<algorithm>
        len = 0;
    }
};

bign change(char str[]) { // 整数转换为 bign
    bign a;
    a.len = strlen(str); // bign 的长度即字符串的长度
    for (int i = 0; i < a.len; i++) {
        a.d[i] = str[a.len - i - 1] - '0'; // 注意，逆着赋值，str[0]是最高位
        return a;
    }

    int compare(bign a, bign b) {
        if (a.len > b.len) return 1;
        else if (a.len < b.len) return -1;
        else {
            for (int i = a.len - 1; i >= 0; i--) { // 从高位到低位比较
                if (a.d[i] > b.d[i]) return 1;
                else if (a.d[i] < b.d[i]) return -1;
            }
            return 0; // 两数相等
        }
    }

    void print(bign a) {
        for (int i = a.len - 1; i >= 0; i--)
            cout << a.d[i];
    }

    bign add(bign a, bign b) { // 高精度 a + b
        bign c; // c = a + b
        int carry = 0; // 进位
        for (int i = 0; i < a.len || i < b.len; i++) {
            int t = a.d[i] + b.d[i] + carry;
            c.d[c.len++] = t % 10;
            carry = t / 10;
        }
        if (carry)
            c.d[c.len++] = carry;
        return c;
    }

    bign sub(bign a, bign b) {
        bign c;
        for (int i = 0; i < a.len || i < b.len; i++) {
            if (a.d[i] < b.d[i]) {
                a.d[i + 1]--;
                a.d[i] += 10;
            }
            c.d[c.len++] = a.d[i] - b.d[i];
        }
        while (c.len - 1 >= 1 && c.d[c.len - 1] == 0) {
            c.len--; // 去除最高位的0，同时至少保留一位最低位
        }
        return c;
    }

    bign multi(bign a, int b) {
        bign c;
        int carry = 0; // 进位
        for (int i = 0; i < a.len; i++) {
            int t = a.d[i] * b + carry;
            c.d[c.len++] = t % 10;
            carry = t / 10;
        }
        while (carry) {
            c.d[c.len++] = carry % 10;
            carry /= 10;
        }
        return c;
    }

    bign divide(bign a, int b int &r) { // 返回商，r为余数
        bign c;
        c.len = a.len;
        for (int i = a.len - 1; i >= 0; i--) { // 从高位开始
            r = r * 10 + a.d[i]; // 加上上一位遗留的余数
            if (r < b) c.d[i] = 0; // 该位不够除，为0
            else { // 够除
                c.d[i] = r / b; // 商
                r = r % b;
            }
        }
        while (c.len - 1 >= 1 && c.d[c.len - 1] == 0) {
            c.len--; // 去除最高位的0，同时至少保留一位最低位
        }
        return c;
    }
```