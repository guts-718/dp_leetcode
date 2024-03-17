
****** ( 9 * 9 > 8 * 8 ) *******
Russian doll, longest increasing subsequence , mountain array questions and this are very similar.
https://leetcode.com/problems/partition-equal-subset-sum/solutions/462699/whiteboard-editorial-all-approaches-explained/
673. Number of Longest Increasing Subsequence

Leetcode questions based on "intervals": C++ Solutions and explanations
56. Merge Intervals

Simple sorting and merging. We keep updating p until there is an overlap and then push it to the final array.

    
vector<vector<int>> merge(vector<vector<int>>& v) {
		int n = v.size();
		sort(v.begin(),v.end());
		vector<vector<int>> ans;
		vector<int> p = v[0];
		for(int i=1;i<n;i++)
		{
			if(p[1] >= v[i][0])
				p = {min(p[0],v[i][0]), max(p[1],v[i][1])};
			else
			{
				ans.push_back(p);
				p = v[i];
			}
		}
		ans.push_back(p);
		return ans;
}
----------------------------------------------------------------------------------------------------------------

435. Non-overlapping Intervals

we have to sort the intervals on the basis of thier end points,
then use a greeady approach to find the answer.

If p is ending after the start of current element, we eliminate the current element but not the element contained in p because the elements are sorted according to their end points and p will have a lesser end point than the current element. So we eliminate current element to reduce the probability of overlapping with next element.

bool comp(vector<int> a, vector<int> b)
{
    if(a[1] == b[1]) return a[0]<b[0];
    return a[1]<b[1];
}

int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(),comp);
        int n = intervals.size();
        int ans = 0;
        vector<int> p = intervals[0];
        
        for(int i=1;i<n;i++)
        {
            if(p[1] > intervals[i][0])
                ans++;
            else
                p = intervals[i];
        }
        return ans;
    }

Why do we sort on the basis of end points, not on start points.

    suppose you have cases like : (1,8), (2,3), (3,4), (5,9)
    if you sort in the basis of start points you will end up considering (1,8) and deleting rest which collide with (1,8).
    For a greedy approach you will want the point with lower end point to be considered.
    But, We can sort on the basis of start point also, just a little tweak in the algorithm will work out that way. In case of overlap, remove the interval having the farther end point.

int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());
        int n = intervals.size();
        int ans = 0;
        int p= intervals[0][1];
        
        for(int i=1;i<n;i++)
        {
            if(p > intervals[i][0])
            {
                ans++;
                p = min(p, intervals[i][1]);
            }
            else
                p = intervals[i][1];
        }
        return ans;
    }
------------------------------------------------------------------------------------------------------

452. Minimum Number of Arrows to Burst Balloons

Very similar to 435. Non-overlapping Intervals but we have to return the number of remaining intervals.(n-ans).

int findMinArrowShots(vector<vector<int>>& points) {
		int n = points.size();
		sort(points.begin(),points.end());
		int p = points[0][1];
		int ans = 0;   
		 for(int i=1;i<n;i++)
		    {
		        if(p >= points[i][0])
		        {
		            ans++;
		            p = min(p, points[i][1]);
		        }
		        else
		            p = points[i][1];
		    }
		
    return n-ans;
}

Meeting rooms II

Here we are using a priority queue so that we can keep track of the earliest ending meeting. If the start of current meeting isn't greater or equal to the end value of top element in the priority queue, we need an extra room to accommodate the meeting and hence we push the element into the priority queue. Otherwise, we pop the top element and push the current element(i.e., we re-use the room). Eventually, the size of priority queue turns out to be our answer.

struct comp {
    bool operator()(vector<int> p1, vector<int> p2)
    {
        return p1[1] > p2[1];
    }
};

int Solution::solve(vector<vector<int> > &v) {
   
        int n = v.size();
        if(n==0) return 0;
        sort(v.begin(),v.end());
        priority_queue<vector<int>, vector<vector<int>>, comp> pq;
        pq.push(v[0]);

        for(int i=1;i<n;i++)
        {
            vector<int> temp = pq.top();
            int x = temp[1];
            if(v[i][0] < x)
            {
                pq.push(v[i]);
            }
            else
            {
                pq.pop();
                pq.push(v[i]);
            }
        }
        return pq.size();
    
}


-------------------------------------------------------------------------------------------------------------------------------------

5. Longest Palindromic Substring -- Given a string s, return the longest palindromic substring in s.
   1 <= s.length <= 1000
*  single loop --> for(i=0 ---> len)
   2 loops nested inside outer for loop --> basically har i ko middle element manke check krna chah rhe -->
   l=i,r=i;
   l=i,r=i+1(yaha +1 instead of -1 liye as i ko ham 0 se start kre the)
        l=i,r=i; // l=i,r=i+1
        while(l>=0 && r<s.length() && s[l]==s[r]){
           if(r-l+1>maxLen){
               temp = s.substr(l,r-l+1);
               maxLen = r-l+1;

           }
           l--;
           r++;
       }

    BRUTE --> O(n^3) --> O(n^2) for each substring and O(n) for pallindrome checking
            for (int i = 0; i < s.length(); ++i) { 
            for (int j = i + max_len; j <= s.length(); ++j) {      //yaha maxLen ka use kiye hai to optimise it a bit
                if (j - i > max_len && isPalindrome(s.substr(i, j - i))) {
                    max_len = j - i;
                    max_str = s.substr(i, j - i);
                }
            }
        } 

    We will iterate over the string and mark the diagonal elements as true as every single character is a palindrome.
    we are using a 2D matrix 

    class Solution {
public:
    string longestPalindrome(string s) {
        if (s == string(s.rbegin(), s.rend())) {
            return s;
        }

        string left = longestPalindrome(s.substr(1));
        string right = longestPalindrome(s.substr(0, s.size() - 1));

        if (left.length() > right.length()) {
            return left;
        } else {
            return right;
        }
    }
};

----------------------------------------------------------------------------------------------------------------------------------------------

10. Regular Expression Matching ----  Given an input string s and a pattern p 
    '.' Matches any single character.​​​​ ( size of s and p <=20)
    '*' Matches zero or more of the preceding element.
        
        bool solve(string &s, string &p, int i, int j){
        if(i >= s.size() && j >= p.size()) return true;
        if(j >= p.size()) return false;

        bool ans = false;
        bool match = ( i < s.size() ) && (s[i] == p[j] || p[j] == '.');

        if(j+1 < p.size() && p[j+1] == '*'){
            ans = ( match && solve(s, p, i+1, j) ) || solve(s, p, i, j+2);
        }
        else if(match){
            ans = solve(s, p, i+1, j+1);
        }

        return ans;
    }

    
    
    class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size(), n = p.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p[j - 1] == '*') {
                    dp[i][j] = dp[i][j - 2] || (i && dp[i - 1][j] && (s[i - 1] == p[j - 2] || p[j - 2] == '.'));
                } else {
                    dp[i][j] = i && dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
                }
            }
        }
        return dp[m][n];
    }
};

-------------------------------------------------------------------------------------------------------------------------------------

22. Generate Parentheses -- Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
    
    class Solution {
public:
    vector<string>result;
    void ans(int first,int last,int n,string temp){
        if(temp.length()==2*n){
            result.push_back(temp);
            return;
        }
        if(first<n)ans(first+1,last,n,temp+"(");      // the order of these 2 statements do not matter
        if(last<first)ans(first,last+1,n,temp+")");

    }
    vector<string> generateParenthesis(int n) {
      ans(0,0,n,"");
      return result;
    }
};

 ( ( ( ( ( (  ---> yaha total 6 ')' jarurat hai aur ham first blank me 1 , 2ns blank me 2 , 3rd me 3 krke bhar skte hai
 meaning a+b+c+d+e+f = 6 (satisfying opening closing)
 above program me the moment hame mauka mil rha ')' add krne ka ham add kr de rhe...

------------------------------------------------------------------------------------------------------------------------------------------

32. Longest Valid Parentheses -- Given a string containing just the characters '(' and ')', return the length of the longest valid (well-formed) parentheses substring
.   class Solution {
public:
    int longestValidParentheses(std::string s) {
        int maxCount = 0;
        std::stack<int> st;
        st.push(-1);

        for (int i = 0; i < s.length(); ++i) {
            if (s[i] == '(') {
                st.push(i);  // Push the index of the opening parenthesis onto the stack
            } else {
                // yaha jo pop kiye aur jo pop kerne ke baad top of stack pe hai wo dono adjacent opening parenthesis hi hoga...
                st.pop();  // Pop the index of the matching opening parenthesis

                //pop krne pr empty ho gya it means string ka start char hi '(' tha
                if (st.empty()) {
                    st.push(i);  // If stack is empty, push the current index (unmatched closing parenthesis)
                } else {
                    maxCount = std::max(maxCount, i - st.top());  // Calculate the length of valid parentheses substring
                }
            }
        }

        return maxCount;
    }
};
-1 tabhi pop hoga jb while iterating no of ')' no of opening parenthesis se jyada rhe  -1(()))

jo index at the top of stack hai is signifying the index(earilest) which does form valid parenthesis with the current ')' brace..
agar start me hi ')' aa gya to -1 ko pop krke iska index daaldenge which works..
opening brace aa gya to to problem hi nhi hai
)() --> 2 answer de rha above program -- islie -1 dale hai
i-st.top() -- kre hai i-st.top()+1 nhi islie st.top() pr hai last char which does not work with the current string
only a ')' can replace -1 ---> agar satck empty ho gya tb hame ek naya -1 ka replacement mil gya hai
agar parenthesis wagera me length ka jarurqat padta hai to hamesha index put karo into stack

----------------------------------------------------------------------------------------------------------------------------------------------------------

45. Jump Game II
    You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].
    Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where:

    class Solution {
public:
    long long int min_jumps(vector<int>& nums, int index,vector<int>&dp){
        if(index>=nums.size()-1)return 0;
        if(dp[index]!=-1)return dp[index];
        long long int mini=INT_MAX;
        for(int i=1;i<=nums[index];i++){
            //yaha 0 se < tk jate to stack overflow bata rha tha as 0 krne pr infinite loop me chala jaega.. even if it cannot be used(ie 0 hai then the condition i<=nums[index] will handle it)
            mini = min(mini,1+min_jumps(nums,index+i,dp));
        }
        return dp[index]=mini;
    }
    int jump(vector<int>& nums) {
        vector<int>dp(nums.size()+1,-1);
        return min_jumps(nums,0,dp);
        
        
    }
};


--------------------------------------------------------------------------------------------------------------------

53. Maximum Subarray -- Given an integer array nums, find the subarray with the largest sum, and return its sum. -->KADANE'S ALGORITHM
    class Solution {
    public:
        int maxSubArray(vector<int>& nums) {
            int MAX = INT_MIN;
            int sum = 0;
            for(int i = 0; i < nums.size(); i++) {
                sum += nums[i];
                MAX = max(sum, MAX);
                if(sum < 0) sum = 0;
            }
            return MAX;
        }
    };

---------------------------------------------------------------------------------------------------------------------------------

55. Jump Game
    You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.
    Return true if you can reach the last index, or false otherwise.
    
class Solution {
public:
    bool ans(vector<int>&nums,int ind,vector<int>&dp){
        if(ind==nums.size()-1)return true;
        if(ind>nums.size())return false;
        if(dp[ind]!=-1)return dp[ind];
        
        //yaha i=0 se initiallise kar dete to dhokha ho jata(stack overflow -- infinite loop)
        for(int i=1;i<=nums[ind];i++){
            if(ans(nums,ind+i,dp))return dp[ind]=true;

        }
        return dp[ind]=false;
    }
    bool canJump(vector<int>& nums) {
        vector<int>dp(nums.size()+1,-1);
        return ans(nums,0,dp);
    }
};


class Solution {
public:
    bool canJump(vector<int>& nums) {
        int n=nums.size(), reachable_So_Far;
    
        for(int i=0;i<n;i++){
            if( reachable_So_Far<i) return false;
            reachable_So_Far=max( reachable_So_Far, i+nums[i]);
            if(reachable_So_Far>=n-1) return true;
        }

          return false;
    }
  
};

    class Solution {
public:
    bool canJump(vector<int>& nums) {
        vector<int>v;
        if(nums.size()==1)return true;
        for(int i=0;i<nums.size();i++){
            if(nums[i]==0)v.push_back(i);
        }
        int cnt=0;
        for(auto x:v){
            for(int i=0;i<x;i++){
                if(nums[i]>(x-i)){
                    cnt++;
                    break;

                }
                else if(x==nums.size()-1 && nums[i]>=(x-i)){
                    cnt++;
                    break;
                }
            }
        }
        if(cnt==v.size())return true;
        else return false;
        
    }
};

-------------------------------------------------------------------------------------------------------------

62. Unique Paths -- [0,0] --> [m-1,n-1] only right and downward movement allowed
    maths --> RRRDD --- for m rows -- m-1 (D) for n columns n-1 (R) --> (m-1 + n-1)! / (m-1)!(n-1)!

// class Solution {
// public:
//     int helper(int m, int n,int ind_x,int ind_y,vector<vector<int>>&dp){
//         if(ind_x<0|| ind_y<0)return 0;
//         else if(ind_x==0 && ind_y==0)return 1;
//         if(dp[ind_x][ind_y]!=-1)return dp[ind_x][ind_y];

//         int right = helper(m,n,ind_x-1,ind_y,dp);
//         int down = helper(m,n,ind_x,ind_y-1,dp);
//         return dp[ind_x][ind_y] = down+right;
//     }
//     int uniquePaths(int m, int n) {
//         vector<vector<int>>dp(m+1,vector<int>(n+1,-1));
//         return helper(m,n,m-1,n-1,dp);
        
//     }
// };



class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 0));
        
        for (int i = 0; i < m; ++i) {
            dp[i][0] = 1;
        }
        for (int j = 0; j < n; ++j) {
            dp[0][j] = 1;
        }
        
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        
        return dp[m-1][n-1];
    }
};


class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 1));
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }
};


class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> pre(n, 1), cur(n, 1);
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                cur[j] = pre[j] + cur[j - 1];
            }
            swap(pre, cur);
        }
        return pre[n - 1];
    }
};


class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> cur(n, 1);
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                cur[j] += cur[j - 1];
            }
        }
        return cur[n - 1];
    }
};



----------------------------------------------------------------------------------------------------------

63. Unique Paths II -- similar as above just bich me 
    An obstacle and space are marked as 1 or 0 respectively in grid. A path that the robot takes cannot include any square that is an obstacle.

    class Solution {
public:
    int helper(int m, int n,int ind_x,int ind_y,vector<vector<int>>&dp,vector<vector<int>>& obstacleGrid){
        if(ind_x<0|| ind_y<0)return 0;
        if(obstacleGrid[ind_x][ind_y]==1)return 0;  //isko as a base case manna chahiye or else dikkat ki gunjais hai
        else if(ind_x==0 && ind_y==0)return 1;
        if(dp[ind_x][ind_y]!=-1)return dp[ind_x][ind_y];

        int right = helper(m,n,ind_x-1,ind_y,dp,obstacleGrid);
        int down = helper(m,n,ind_x,ind_y-1,dp,obstacleGrid);
        return dp[ind_x][ind_y] = down+right;
    }

    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m=obstacleGrid.size();
        int n=obstacleGrid[0].size();
         vector<vector<int>>dp(m+1,vector<int>(n+1,-1));
        return helper(m,n,m-1,n-1,dp,obstacleGrid);
        
    }
};

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

64. Minimum Path Sum
--> Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.
    Note: You can only move either down or right at any point in time
ans -> if(ind_x<0||ind_y<0)return 1e8;

class Solution {
public:
    int helper(int ind_x,int ind_y,vector<vector<int>>& grid,vector<vector<int>>& dp){
        if(ind_x<0||ind_y<0)return 1e8;
        if(ind_x==0 && ind_y==0)return grid[0][0];
        if(dp[ind_x][ind_y]!=-1)return dp[ind_x][ind_y];

        int right = helper(ind_x-1,ind_y,grid,dp);
        int up = helper(ind_x,ind_y-1,grid,dp);
        return dp[ind_x][ind_y]=grid[ind_x][ind_y]+min(right,up);
    }
    int minPathSum(vector<vector<int>>& grid) {
        int m=grid.size();
        int n=grid[0].size();
        vector<vector<int>>dp(m+1,vector<int>(n+1,-1));
        return helper(m-1,n-1,grid,dp);
        
    }
};

   class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size(); 
        vector<vector<int> > sum(m, vector<int>(n, grid[0][0]));
        for (int i = 1; i < m; i++)
            sum[i][0] = sum[i - 1][0] + grid[i][0];
        for (int j = 1; j < n; j++)
            sum[0][j] = sum[0][j - 1] + grid[0][j];
        for (int i = 1; i < m; i++)
            for (int j = 1; j < n; j++)
                sum[i][j]  = min(sum[i - 1][j], sum[i][j - 1]) + grid[i][j];
        return sum[m - 1][n - 1];
    }
};


70. Climbing Stairs
    You are climbing a staircase. It takes n steps to reach the top.
    Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
---> f(n)=f(n-1)+f(n-2)  ----> same as fibonacci

     class Solution {
public:
    int climbStairs(int n) {
        int pre1=1;
        int pre2=1;
        int curr=0;
        if(n==0||n==1)return 1;
        for(int i=2;i<=n;i++){
            curr=pre1+pre2;
            pre2=pre1;
            pre1=curr;
            
        }
        return curr;
    }
};


------------------------------------------------------------------------------------------------------------------------

72. Edit Distance
    Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.
    we can insert , delete and replace a character

    class Solution {
public:
    int helper(int i,int j, string word1, string word2,vector<vector<int>>&dp){
        if(i<0)return j+1;
        if(j<0)return i+1;
        if(dp[i][j]!=-1)return dp[i][j];
        
        if(word1[i]==word2[j])return 0+helper(i-1,j-1,word1,word2,dp);
        int l = helper(i,j-1,word1,word2,dp);
        int r = helper(i-1,j,word1,word2,dp);
        int h = helper(i-1,j-1,word1,word2,dp);
        return dp[i][j] = 1 + min(l,min(r,h));
    }
    int minDistance(string word1, string word2) {
        int m=word1.size();
        int n=word2.size();
        vector<vector<int>>dp(m,vector<int>(n,-1));
        return helper(m-1,n-1,word1,word2,dp);

        
    }
};

class Solution {
public:
    int minDistance(string word1, string word2) {
        vector<vector<int>>dp(word1.length()+1,vector<int>(word2.length()+1,0));

        // agar word2.length()=0 hai to answer=word.length() ie itna no of insertion hoga
        for(int i=1;i<=word1.size();i++){
            dp[i][0]=i;
        }
        // agar word1.length()=0 hai to answer=word2.length() ie itna no of deletion hoga
        for(int i=1;i<=word2.length();i++){
            dp[0][i]=i;
        }
        for(int i=1;i<=word1.size();i++){
            for(int j=1;j<=word2.length();j++){
                if(word1[i-1]==word2[j-1])dp[i][j]=dp[i-1][j-1];
                else{
                    dp[i][j]=min(dp[i][j-1],min(dp[i-1][j],dp[i-1][j-1]))+1;
                    }
      
            }
        }
        return dp[word1.length()][word2.length()];

    }
};

--------------------------------------------------------------------------------------------------------------------------------------
85. MAXIMUM RECTANGLE
Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.


class Solution {
public:
    int maximalRectangle(vector<vector<char>>& mat) {
        int m = mat.size(); //rows
        if(m == 0)
            return 0;
        int n = mat[0].size(); //cols
        int ans = 0;

        // [i and j] heights of 1's ke lie hai aur vector v length of such 1's ke lie hai
        for(int i=0;i<m;i++)
        {
            vector<int>v(n,1);
            for(int j=i;j<m;j++)
            {
                // [i,j] me kis kis column me all 1's hai
                for(int k=0;k<n;k++)v[k] = v[k]&mat[j][k];
                int len = 0;
                int max_len = 0;

                // to calculate max length(of all 1's in a row)
                for(int k=0;k<n;k++)
                {
                    if(v[k] == 0)
                        len = 0;
                    else
                        len++;
                    max_len = max(max_len,len);
                }
                // [i,j] me kya answer hai {n^2 loop hai so sare hi possible [i,j] range ko cover kr le rhe}
                ans = max(ans,max_len*(j-i+1));
            }
        }
        return ans;
    }
};
-------------------------------------------------------------------------------------------------------------------------

87. Scramble String

class Solution {
public:
    unordered_map<string,bool> mp;
    
    bool isScramble(string s1, string s2) {
        int n = s1.size();
        if(s1==s2) return true;   
        if(n==1) return false;
        
        string key = s1+" "+s2;
        
        if(mp.find(key)!=mp.end()) return mp[key];

        for(int i=1;i<n;i++)
        {
            if(isScramble(s1.substr(0,i),s2.substr(0,i)) && 
                            isScramble(s1.substr(i),s2.substr(i)))
                return mp[key] = true;
            
            if(isScramble(s1.substr(0,i),s2.substr(n-i)) &&
                        isScramble(s1.substr(i),s2.substr(0,n-i)))
                return mp[key] = true;
        }
        
        return mp[key] = false;
    }
};



---------------------------------------------------------------------------------------------------------------------------------------------

91. Decode Ways

class Solution {
public:
    // if there are leading zeros then return 0 or if 0 comes while  if(s[ind]!='0')result+= helper(ind+1,s,dp);
    int helper(int ind,string s,vector<int>&dp){
        if(ind>=s.length())return 1;
        
        //in order to skip this line we are using ind+2 in the second recusive line
        if(s[ind]=='0')return 0;
        if(dp[ind]!=-1)return dp[ind];
        int result = 0;
        if(s[ind]!='0')result+= helper(ind+1,s,dp);
        // if "10" -- upper line will go ind+1 and then when s[ind]==0 we'll get 0 from above line
        
        // since below line goes ind+2 it will give us the result
        //below line deals with both zeros and where 2 adj digits can form 2 results
        // below line (s[i+1]=='0') ko skip krne ke kaam aata hai
        if(ind+1<s.length() && (s[ind]=='2' && s[ind+1]<='6' || s[ind]=='1') )result+=helper(ind+2,s,dp);
        
        return dp[ind]=result;
    }
    int numDecodings(string s) {
        vector<int>dp(s.length(),-1);
        return helper(0,s,dp);
        
    }
};

--------------------------------------------------------------------------------------------------------------------------------------------------------------
95. Unique Binary Search Trees II
    Given an integer n, return all the structurally unique BST's (binary search trees), which has exactly n nodes of unique values from 1 to n. Return the answer in any order.


//Approach-2 Recursion + Memo - ACCEPTED
class Solution {
public:
    
    map<pair<int, int>, vector<TreeNode*>> mp;
    
    vector<TreeNode*> solve(int start, int end) {
        
        if(start > end) {
            return {NULL};
        }
        
        if(start == end) {
            TreeNode* root = new TreeNode(start);
            return {root};
        }
        
        if(mp.find({start, end}) != mp.end())
            return mp[{start, end}];
        
        vector<TreeNode*> result;
        for(int i = start; i <= end; i++) {
            
            vector<TreeNode*> leftList  = solve(start, i-1);
            vector<TreeNode*> rightList = solve(i+1, end);
            
            for(TreeNode* leftRoot : leftList) {
                
                for(TreeNode* rightRoot : rightList) {
                    
                    TreeNode* root = new TreeNode(i);
                    root->left  = leftRoot;
                    root->right = rightRoot;
                    
                    result.push_back(root);
                    
                }
                
            }
            
        }
        
        return mp[{start, end}] = result;
        
    }
    
    vector<TreeNode*> generateTrees(int n) {
        return solve(1, n);
    }
};
-------------------------------------------------------------------------------------------------------------------------------------------------
96.  Unique Binary Search Trees
---> Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n
---> class Solution {
public:
    long CalculateCoeff(int n,int k) //Function to calculate Ci(n.k)
    {
        long res=1;
        if(k>n-k)
            k=n-k;                    //Since Ci(n,k)=Ci(n,n-k), property of binomial coefficients
        for(int i=0;i<k;i++)
        {
            res*=(n-i);
            res/=(i+1);
        }
        return res;
    }
    int numTrees(int n) {
        return CalculateCoeff(2*n,n)/(n+1);
    }
};

------------------------------------------------------------------------------------------------------------------------------------------------------------------
97 INTERLEAVING STRING --
.Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.


class Solution {
public:
    bool rec(string s1,string s2,string s3,int i,int j,int k,vector<vector<int>>&dp){
        if(k==s3.size()&&i==s1.size()&&j==s2.size())return true;
        
        if(i>s1.size()||j>s2.size())return false;
        
        if(dp[i][j]!=-1)return dp[i][j];
        
        if(s3[k]==s2[j]&&s3[k]==s1[i]){
            return dp[i][j]= rec(s1,s2,s3,i+1,j,k+1,dp)||rec(s1,s2,s3,i,j+1,k+1,dp);
        }
        else if(s1[i]==s3[k]){
            return dp[i][j]= rec(s1,s2,s3,i+1,j,k+1,dp);
        }
        else if(s3[k]==s2[j]){
            return dp[i][j]= rec(s1,s2,s3,i,j+1,k+1,dp);
        }
        else{
            return dp[i][j]= false;
        }
    }
    bool isInterleave(string s1, string s2, string s3) {
        vector<vector<int>>dp(s1.size()+1,(vector<int>(s2.size()+1,-1)));
        return rec(s1,s2,s3,0,0,0,dp);
    }
};
because in else if cond u r not storing as it will be out of bounds if u store data lets run for 1st test case in description at first s2 will be completed now j=-1 and a=1 if u store dp[1][-1] it goes out of bound so just store when only i>=0 && j>=0
hope u got my point

see in the 2nd and 3rd case we are not checking that i-1 or j-1 do exist but we call the function on it and so it may go out of bounds. Now, the thing is we are not checking the same in the first case too, but you see that in first case we have an or operation on 2 calls so one them will atleast give an ans.

You can avoid runtime error by shifting the indices of dp array by 1.
declaration: vector<vector> dp(s1.size()+2, (vector(s2.size()+2, -1)));
and use dp[i+1][j+1] in place of dp[i][j]


class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        vector<vector<int>>dp(s1.size()+1,(vector<int>(s2.size()+1,0)));
        if(s3.size()!=s1.size()+s2.size()){
            return false;
        }
        for(int i=s1.size();i>=0;i--){
            for(int j=s2.size();j>=0;j--){
                int k=i+j;
                if(i==s1.size()&&j==s2.size()){
                    dp[i][j]=1;
                }
                else if(s3[k]==s2[j]&&s3[k]==s1[i]){
                    dp[i][j]= dp[i+1][j]||dp[i][j+1];
                }
                else if(s1[i]==s3[k]){
                    dp[i][j]= dp[i+1][j];
                }
                else if(s3[k]==s2[j]){
                    dp[i][j]= dp[i][j+1];
                }
                else{
                    dp[i][j]= false;
                }  
            }
        }
        return dp[0][0];
    }
};




-------------------------------------------------------------------------------------

115. Distinct Subsequences
     Given two strings s and t, return the number of distinct subsequences of s which equals t.
     The test cases are generated so that the answer fits on a 32-bit signed integer.
   
    class Solution {
public:
    int helper(int i,int j,string s, string t,vector<vector<int>>&dp){
        if(j<0)return 1;
        if(i<0)return 0;
        if(dp[i][j]!=-1)return dp[i][j];
        int result=0;

        if(s[i]==t[j]) result= helper(i-1,j,s,t,dp)+helper(i-1,j-1,s,t,dp);
        else result = helper(i-1,j,s,t,dp);
        return dp[i][j]=result;
    }
    int numDistinct(string s, string t) {
        int m=s.length();
        int n=t.length();
        vector<vector<int>>dp(m,vector<int>(n,-1));
        return helper(m-1,n-1,s,t,dp);

        
    }
};




Well, a dynamic programming problem. Let's first define its state dp[i][j] to be the number of distinct subsequences of t[0..i - 1] in s[0..j - 1]. Then we have the following state equations:

    General case 1: dp[i][j] = dp[i][j - 1] if t[i - 1] != s[j - 1];
    General case 2: dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1] if t[i - 1] == s[j - 1];
    Boundary case 1: dp[0][j] = 1 for all j;
    Boundary case 2: dp[i][0] = 0 for all positive i.

Now let's give brief explanations to the four equations above.

    If t[i - 1] != s[j - 1], the distinct subsequences will not include s[j - 1] and thus all the number of distinct subsequences will simply be those in s[0..j - 2], which corresponds to dp[i][j - 1];
    If t[i - 1] == s[j - 1], the number of distinct subsequences include two parts: those with s[j - 1] and those without;
    An empty string will have exactly one subsequence in any string :-)
    Non-empty string will have no subsequences in an empty string.

Putting these together, we will have the following simple codes (just like translation :-)):

class Solution {
public:
    int numDistinct(string s, string t) {
        int m = t.length(), n = s.length();
        vector<vector<int>> dp(m + 1, vector<int> (n + 1, 0));
        for (int j = 0; j <= n; j++) dp[0][j] = 1;
        for (int j = 1; j <= n; j++)
            for (int i = 1; i <= m; i++)
                dp[i][j] = dp[i][j - 1] + (t[i - 1] == s[j - 1] ? dp[i - 1][j - 1] : 0);
        return dp[m][n];
    }
};  


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

118. Pascal's Triangle

int nCr(int n, int r,vector<vector<int>>&dp)
{
    if (r > n)
        return 0;
    if (r == 0 || r == n)
        return 1;
    if(dp[n][r]!=-1)return dp[n][r];
    return dp[n][r]=nCr(n - 1, r - 1,dp) + nCr(n - 1, r,dp);
}

    vector<vector<int>> generate(int numRows) {
        vector<vector<int>>ans;
        for(int i=1;i<=numRows;i++){
            vector<int>temp;
            for(int j=1;j<=i;j++){     /// ith row me i elements hai...

                int x = recur(i-1,j-1);.   /// (i-1) C (r-1)
                temp.push_back(x);
                
            }
            ans.push_back(temp);
        }
        return ans;
        
    }

--------------------------------------------------------------------------------------------------

119. Pascal's Triangle II --- return the rowIndexth (0-indexed) row of the Pascal's triangle.
----> (i-1) C (r-1) ===> 1 indexed {r=1 ---> i} {ith row mw i elements hote hai}
  for(int i=1;i<=numRows+1;i++){
            int x = recur(numRows,i-1);
            temp.push_back(x);
            
        }

------------------------------------------------------------------------------------------------------------

120. Triangle --- Given a triangle array, return the minimum path sum from top to bottom.
     if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.
---> single origin multiple destination type problem hai to isme origin se hi start krenge -->

// class Solution {
//    public:
//     int dfs(int i, int j, int n, vector<vector<int>>& triangle, vector<vector<int>>& memo) {
//         if (i == n) return 0;
//         if (memo[i][j] != -1) return memo[i][j];
        
//         int lower_left = triangle[i][j] + dfs(i + 1, j, n, triangle, memo);
//         int lower_right = triangle[i][j] + dfs(i + 1, j + 1, n, triangle, memo);
//         memo[i][j] = min(lower_left, lower_right);
        
//         return memo[i][j];
//     }
//     int minimumTotal(vector<vector<int>>& triangle) {
//         int n = triangle.size();
//         vector<vector<int>> memo(n, vector<int>(n, -1));
//         return dfs(0, 0, n, triangle, memo);
//     }
// };


class Solution {
   public:
    int dfs(int i, int j, int n, vector<vector<int>>& triangle, vector<vector<int>>& memo) {
        if (i<0 || j<0) return 1e8;
        if(i==0 && j==0)return triangle[0][0];
        if (memo[i][j] != -1) return memo[i][j];
        int lower_left=1e8,lower_right=1e8;
        if(j<triangle[i].size())lower_left = triangle[i][j] + dfs(i - 1, j, n, triangle, memo);
        if(j<triangle[i].size())lower_right = triangle[i][j] + dfs(i - 1, j - 1, n, triangle, memo);
        memo[i][j] = min(lower_left, lower_right);
        
        return memo[i][j];
    }
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size();
        int m=triangle[n-1].size();
        vector<vector<int>> memo(n+1, vector<int>(m+1, -1));
        int mini=INT_MAX;
        for(int i=0;i<m;i++){
            mini=min(mini,dfs(n-1, i, n, triangle, memo));

        }
        return mini;
         
    }
};

--------------------------------------------------------------------------------------------------------------------------
121. Best Time to Buy and Sell Stock {single buy single sell}



class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int mini=prices[0];
        int max_p=0;
        for(int i=1;i<prices.size();i++){
            if(prices[i]<mini)mini=prices[i];
            max_p = max(max_p,prices[i]-mini);
        }
        return max_p;

    }
};


--------------------------------------------------------------------------------------------------------------------

122. Best Time to Buy and Sell Stock II
    On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.
    Find and return the maximum profit you can achieve.

class Solution {
public:
    int helper(vector<int>&prices,int index,int buy,vector<vector<int>>&dp){
        if(index>=prices.size())return 0;
        int max_p = 0;
        if(dp[index][buy]!=-1)return dp[index][buy];
        if(buy){
            max_p =  max(-prices[index]+helper(prices,index+1,0,dp),helper(prices,index+1,1,dp));
        }
        else{
          max_p =  max(prices[index]+helper(prices,index+1,1,dp),helper(prices,index+1,0,dp));
        }
        return dp[index][buy] =  max_p;


    }
    int maxProfit(vector<int>& prices) {
        vector<vector<int>>dp(prices.size(),vector<int>(2,-1));
        return helper(prices,0,1,dp);

        
    }
};


--------------------------------------------------------------------------------------------------------------------------------------
123. Best Time to Buy and Sell Stock III
     You are given an array prices where prices[i] is the price of a given stock on the ith day. Find the maximum profit you can achieve. You may complete at most two transactions.
     Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

class Solution {
public:
    int helper(vector<int>& prices,vector<vector<vector<int>>>&dp,int index,int buy,int cnt){
        if(index==prices.size() || cnt==0)return 0;
        if(dp[index][buy][cnt]!=-1)return dp[index][buy][cnt];
        int profit = 0;
        if(buy==1){
            profit = max(-prices[index]+helper(prices,dp,index+1,0,cnt),helper(prices,dp,index+1,1,cnt));

        }
        if(buy==0){
            profit = max(prices[index]+helper(prices,dp,index+1,1,cnt-1),helper(prices,dp,index+1,0,cnt));
        }
        return dp[index][buy][cnt]=profit;
    }
    int maxProfit(vector<int>& prices) {
        vector<vector<vector<int>>>dp(prices.size(),vector<vector<int>>(2,vector<int>(3,-1)));
        return helper(prices,dp,0,1,2);
        //  return dp[0][1][k];
        
    }
};

--------------------------------------------------------------------------------------------------------------------------------------
124. Binary Tree Maximum Path Sum
class Solution {
public:
    int helper(TreeNode* root,int &maxi){
        if(root==NULL)return 0;
        int l=max(0,helper(root->left,maxi));
        int r=max(0,helper(root->right,maxi));
        // maxi is storing the answer for -- root is in the answer(root+ left ST + right ST)
        maxi=max(maxi,root->val+l+r);

        //hamko uper bhejna hai max(left ST,right ST) + node value
        return max(l,r)+root->val;
    }
    int maxPathSum(TreeNode* root) {
        
        int maxi=INT_MIN;
        helper(root,maxi);
        return maxi;
        
    }
};


 return max(l,r)+root->val; ----> ek branch ka value chahie tha islie aisa kiye
 return l+r+root->val ........... agar subtree ka sum chahhie hota to





--------------------------------------------------------------------------------------------------------------------------------------
543. Diameter of Binary Tree

class Solution {
public:
    int diameter(TreeNode* root,int& ans){
        if(root==NULL)return 0;
        int l=diameter(root->left,ans);
        int r=diameter(root->right,ans);

        // niche l+r kare instead of r+l+1 as ham length ko consider kr rhe..not nodes
        ans=max(left+right,ans);
        return max(l,r)+1;
    }
    int diameterOfBinaryTree(TreeNode* root) {
        int ans=0;
        diameter(root,ans);
        return ans;
        
    }
};

--------------------------------------------------------------------------------------------------------------------------------------------------

SUBSTRING ME DP ---->

131. Palindrome Partitioning --->
     Given a string s, partition s such that every substring of the partition is a palindrome Return all possible palindrome partitioning of s.
     1 <= s.length <= 16
    class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> pars;
        vector<string> par;
        partition(s, 0, par, pars);
        return pars;
    }
private: 
    void partition(string& s, int start, vector<string>& par, vector<vector<string>>& pars) {
        int n = s.length();
        if (start == n) {
            pars.push_back(par);
        } else {
            for (int i = start; i < n; i++) {
                if (isPalindrome(s, start, i)) {
                    par.push_back(s.substr(start, i - start + 1));
                    partition(s, i + 1, par, pars);
                    par.pop_back();
                }
            }
        }
    }
    
    bool isPalindrome(string& s, int l, int r) {
        while (l < r) {
            if (s[l++] != s[r--]) {
                return false;
            }
        }
        return true;
    }
};


-------------------------------------------------------------------------------------------------
132. Palindrome Partitioning II
     Given a string s, partition s such that every substring of the partition is a
     palindrome Return the minimum cuts needed for a palindrome partitioning of s.

  class Solution {
public:
    bool ispalindrome(string &s,int i, int j){  
        while(i<j){
            if(s[i]!=s[j]){
                return false;
            }
            i++;
            j--;
        }
        return true;
    }
    
    int minCut(string s) {
        int n=s.length();
        vector<int>dp(n+1,0);
        for(int i=n-1;i>-1;i--){
            int mini=INT_MAX;
            for(int j=i;j<n;j++){
                if(ispalindrome(s,i,j)){
                    int part=1+dp[j+1];
                    mini=min(mini,part);
                }}
            dp[i]=mini;
        }

        return dp[0]-1;
    }
};
  
  
    class Solution {
public:
    int minCut(string s) {
        int n = s.size();
        vector<int> cut(n+1, 0);  // number of cuts for the first k characters
        for (int i = 0; i <= n; i++) cut[i] = i-1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; i-j >= 0 && i+j < n && s[i-j]==s[i+j] ; j++) // odd length palindrome
                cut[i+j+1] = min(cut[i+j+1],1+cut[i-j]);

            for (int j = 1; i-j+1 >= 0 && i+j < n && s[i-j+1] == s[i+j]; j++) // even length palindrome
                cut[i+j+1] = min(cut[i+j+1],1+cut[i-j+1]);
        }
        return cut[n];
    }
};


class RecMemoI {
private:
    int n;
    int memo[2000];
    bool isPalindrome(string& s, int l, int r)
    {
        while(l < r)
        {
            if(s[l++] != s[r--])
            {
                return false;
            }
        }
        return true;
    }
    int solve(string& s, int idx)
    {
        if(idx >= n)
        {
            return 0;
        }
        if(memo[idx] != -1)
        {
            return memo[idx];
        }
        int minSteps = INT_MAX;
        for(int k=idx; k<n; k++)
        {
            if(isPalindrome(s, idx, k))
            {
                int steps = 1 + solve(s, k+1);
                minSteps = min(minSteps, steps);
            }
        }
        return memo[idx] = minSteps;
    }
public:
    int minCut(string s) {
        n = s.size();
        memset(memo, -1, sizeof(memo));
        return solve(s, 0) - 1;
    }
};

--------------------------------------------------------------------------------------------------------------------------
1278. Palindrome Partitioning III
      You are given a string s containing lowercase letters and an integer k. You need to :
      First, change some characters of s to other lowercase English letters.
      Then divide s into k non-empty disjoint substrings such that each substring is a palindrome.
      Return the minimal number of characters that you need to change to divide the string.


class Solution {
public:
	vector<vector<int>> cost;
	int costFun(string &s,int l,int r){
		if(l>=r) return 0;
		if(cost[l][r]!=-1) return cost[l][r];
		return cost[l][r] = (s[l]!=s[r]) + costFun(s,l+1,r-1);
	}

	vector<vector<int>>dp;
	int solve(string &s,int k,int pos,int n){
		if(k==0) return costFun(s,pos,n-1);
		if(pos>=n) return INT_MAX;
		if(dp[pos][k]!=-1) return dp[pos][k];
		int ans=1e6;
		for(int i=pos;i<n-1;i++){
			ans=min(ans,costFun(s,pos,i)+solve(s,k-1,i+1,n));
		}
		return dp[pos][k] = ans;
	}
	int palindromePartition(string s, int k) {
		int n=s.size();
		cost.assign(n,vector<int>(n,-1));
		dp.assign(n,vector<int>(k+1,-1));
		return solve(s,k-1,0,n);
	}
};

















--------------------------------------------------------------------------------------------------------------------------------------------------
1745. Palindrome Partitioning IV

class Solution {
public:
	vector<int> dp;
	vector<vector<int>> pali;
	int isPali(string &s,int l,int r){
		if(l>=r) return 1;
		if(pali[l][r]!=-1) return pali[l][r];
		if(s[l]==s[r]) return pali[l][r] = isPali(s,l+1,r-1);
		return pali[l][r] = 0;
	}

	bool checkPartitioning(string s) {
		int n=s.size();
		pali.assign(n,vector<int>(n,-1));
		for(int i=1;i<n-1;i++){
			for(int j=1;j<=i;j++){
				if(isPali(s,0,j-1) && isPali(s,j,i) && isPali(s,i+1,n-1)) return true;
			}
		}
		return false;
	}
};

-----------------------------------------------------------------------------------------------------------------------------------------

139. Word Break --> Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.
     Note that the same word in the dictionary may be reused multiple times in the segmentation.


class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string>word_set(wordDict.begin(),wordDict.end());
        int n=s.size();
        vector<bool>dp(n+1,0);
        dp[0]=1;
        for(int i=0;i<n;i++){

            if(!dp[i])continue;
            
            for(int j=i+1;j<=n;j++){
                if( word_set.count(s.substr(i,j-i)))
                dp[j]=1;
            }
        }
        return dp[n];
    }
};

class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        vector<bool>dp(s.size(),false);
        dp[0]=true;
        
        for(int i = 0; i <= s.size(); i++)
        {
            for(auto str: wordDict)
            {
                if(dp[i])
                {
                    if(s.substr(i,str.size()).compare(str)==0)
                    {
                        dp[i+str.size()]=true;
                    }
                }
            }
        }return dp[s.size()];
        
    }
};



class Solution {
private:
    bool wordBreak(string s, unordered_set<string> &set, vector<int> &memo, int start){
        if(start == s.size()){
            return true;
        }
        if(memo[start] != -1){
            return memo[start];
        }
        for(int i=start; i<s.size(); i++){
            if(set.count(s.substr(start, i+1-start)) && wordBreak(s, set, memo, i+1)){
                memo[start] = true;
                return true;
            }
        }
        return memo[start] = false;
    }
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        vector<int> memo(s.size(), -1);
        unordered_set<string> set(wordDict.begin(), wordDict.end());
        return wordBreak(s, set, memo, 0);
    }
};




/*

    Time Complexity : O(2^N), Given a string of length N, there are N+1 ways to split it into two parts. At each
    step, we have a choice: to split or not to split. In the worse case, when all choices are to be checked, that
    results in O(2^N).

    Space Complexity : O(N), The depth of the recursion tree can go upto N.
    
    Solved using String + Backtracking + Hash Table.



*************************************** Approach 1 First Code *************************************

class Solution {
private:
    bool wordBreak(string s, unordered_set<string> &set){
        if(s.size() == 0){
            return true;
        }
        for(int i=0; i<s.size(); i++){
            if(set.count(s.substr(0, i+1)) && wordBreak(s.substr(i+1), set)){
                return true;
            }
        }
        return false;
    }
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> set(wordDict.begin(), wordDict.end());
        return wordBreak(s, set);
    }
};







    Time Complexity : O(2^N), Given a string of length N, there are N+1 ways to split it into two parts. At each
    step, we have a choice: to split or not to split. In the worse case, when all choices are to be checked, that
    results in O(2^N).

    Space Complexity : O(N), The depth of the recursion tree can go upto N.
    
    Solved using String + Backtracking + Hash Table.



*************************************** Approach 1 Second Code *****************************************
class Solution {
private:
    bool wordBreak(string s, unordered_set<string> &set, int start){
        if(start == s.size()){
            return true;
        }
        for(int i=start; i<s.size(); i++){
            if(set.count(s.substr(start, i+1-start)) && wordBreak(s, set, i+1)){
                return true;
            }
        }
        return false;
    }
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> set(wordDict.begin(), wordDict.end());
        return wordBreak(s, set, 0);
    }
};






    Time Complexity : O(N^3), Size of recursion tree can go up to N^2.

    Space Complexity : O(N), The depth of the recursion tree can go upto N.
    
    Solved using String + DP(Memoisation) + Hash Table.



***************************************** Approach 2 ****************************************

class Solution {
private:
    bool wordBreak(string s, unordered_set<string> &set, vector<int> &memo, int start){
        if(start == s.size()){
            return true;
        }
        if(memo[start] != -1){
            return memo[start];
        }
        for(int i=start; i<s.size(); i++){
            if(set.count(s.substr(start, i+1-start)) && wordBreak(s, set, memo, i+1)){
                memo[start] = true;
                return true;
            }
        }
        return memo[start] = false;
    }
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        vector<int> memo(s.size(), -1);
        unordered_set<string> set(wordDict.begin(), wordDict.end());
        return wordBreak(s, set, memo, 0);
    }
};








    Time Complexity : O(N^3), There are two nested loops, and substring computation at each iteration. Overall
    that results in O(N^3) time complexity.

    Space Complexity : O(N), Length of dp array is N+1.
    
    Solved using String + DP(Tabulation) + Hash Table.




**************************************** Approach 3 *****************************************

class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        vector<bool> dp(s.size()+1, 0);
        dp[0] = true;
        unordered_set<string> set(wordDict.begin(), wordDict.end());
        for(int i=1; i<=s.size(); i++){
            for(int j=0; j<i; j++){
                if(dp[j] && set.count(s.substr(j, i-j))){
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.size()];
    }
};


-----------------------------------------------------------------------------------------------------------------------

140. Word Break II

 bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string>word_set(wordDict.begin(),wordDict.end());
        int n=s.size();
        vector<bool>dp(n+1,0);
        dp[0]=1;
        for(int i=0;i<n;i++){
            if(!dp[i])continue;
            for(int j=i+1;j<=n;j++){
                if( word_set.count(s.substr(i,j-i)))
                dp[j]=1;
            }
        }
        return dp[n];
    }

******************************** WORD BREAK -2 ****************************************
class Solution {
    public:
vector<string> wordBreak(string s, vector<string>& wordDict) {
        int n=s.size();
        unordered_set<string>word_Set(wordDict.begin(),wordDict.end());
         vector<vector<string>>dp(n+1,vector<string>());
         dp[0].push_back("");
    
          for(int i = 0; i < n; ++i){
            for(int j = i+1; j <= n; ++j){
                string temp = s.substr(i, j-i);
                if(word_Set.count(temp)){
                    for(auto x : dp[i]){
                        dp[j].emplace_back(x + (x == "" ? "" : " ") + temp);  
                    }
                }
            }
        }
         return dp[n];
    }
};


    class Solution {
public:
    vector<string> findAllConcatenatedWordsInADict(vector<string>& words) {
        unordered_set<string>word_set(words.begin(),words.end());
        vector<string>ans;
        for(auto w:words){
            int n=w.size();
            vector<bool>dp(n+1,false);
            dp[0]=1;
            for(int i=0;i<n;i++){
                if(!dp[i])continue;
                for(int j=i+1;j<=n;j++){
                    if(j-i<n and word_set.count(w.substr(i,j-i)))
                    dp[j]=1;
                }
            }
            if(dp[n]==1)ans.push_back(w);
        }
       return ans;
    }
};





class Solution {
    public:
        vector<string> wordBreak(string s, vector<string>& wordDict) 
        {
            int max_len = 0;
            unordered_set<string> dict;
            for(string& str : wordDict)
            {
                dict.insert(str);
                max_len = max(max_len, (int)str.length());
            }

            unordered_map<int, vector<string>> mp;
            return break_word(s, 0, dict, max_len, mp);
        }

    protected:
        vector<string> break_word(  const string& s, int n, unordered_set<string>& dict, 
                                    int max_len, unordered_map<int, vector<string>>& mp)
        {
            if(mp.count(n)) return mp[n];

            string str;
            for(int i = n; i < s.length() && str.length() <= max_len; ++i)
            {
                str += s[i];
                if(dict.count(str))
                {
                    if(i == s.length()-1)
                        mp[n].push_back(str);
                    else
                    {
                        vector<string> vs = break_word(s, i+1, dict, max_len, mp);
                        for(auto& a : vs) mp[n].push_back(str + " " + a);
                    }
                }
            }
            return mp[n];
        }
};


-----------------------------------------------------------------------------------------------------------------------

198. House Robber -->
     You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.


class Solution {
public:
    int helper(vector<int>&nums, int index, vector<int>&dp){
        if(index>=nums.size())return 0;
        if(dp[index]!=-1)dp[index];
        int l=nums[index]+helper(nums,index+2,dp);
        int r=helper(nums,index+1,dp);
        return dp[index]=max(l,r);
    }
    int rob(vector<int>& nums) {
         vector<int>dp(nums.size(),-1);
         helper(nums,0,dp);
         return dp[0];
        
    }
};

class Solution {
public:
    int rob(vector<int>& nums) {
        if(nums.size()==0)return 0;
        int n=nums.size();
        vector<int> dp(n+1,0);
        dp[1]=nums[0];
        for(int i=2;i<=n;i++){
            dp[i]=max(dp[i-1],dp[i-2]+nums[i-1]);
        }
        return dp[n];
    }
};


class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        vector<bool>dp(s.size(),false);
        dp[0]=true;
        
        for(int i = 0; i <= s.size(); i++)
        {
            for(auto str: wordDict)
            {
                if(dp[i])
                {
                    if(s.substr(i,str.size()).compare(str)==0)
                    {
                        dp[i+str.size()]=true;
                    }
                }
            }
        }return dp[s.size()];
        
    }
};


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

140. Word Break II
     Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.
     Note that the same word in the dictionary may be reused multiple times in the segmentation.
class Solution {
public:
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        //insert all the words in the set
        unordered_set<string> set;
        vector<string> res;
        for(auto word:wordDict)
            set.insert(word);
        //to store the current string 
        string curr="";
        findHelper(0,s,curr,set,res);
        return res;
    }
    
    void findHelper(int ind,string s,string curr,unordered_set<string> set,vector<string>& res)
    {
        if(ind==s.length())
        {
            //we have reached end
            curr.pop_back(); //remove the trailing space
            res.push_back(curr);
        }
        string str="";
        for(int i=ind;i<s.length();i++)
        {
            //get every substring and check if it exists in set
            str.push_back(s[i]);
            if(set.count(str))
            {
                //we have got a word in dict 
                //explore more and get other substrings
                findHelper(i+1,s,curr+str+" ",set,res);
            }
        }
    }
};


---------------------------------------------------------------------------------------------------------------------------
WORD BREAK - 1

 bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string>word_set(wordDict.begin(),wordDict.end());
        int n=s.size();
        vector<bool>dp(n+1,0);
        dp[0]=1;
        for(int i=0;i<n;i++){
            if(!dp[i])continue;
            for(int j=i+1;j<=n;j++){
                if( word_set.count(s.substr(i,j-i)))
                dp[j]=1;
            }
        }
        return dp[n];
    }
     
BREAK 2

vector<string> wordBreak(string s, vector<string>& wordDict) {
        int n=s.size();
        unordered_set<string>word_Set(wordDict.begin(),wordDict.end());
         vector<vector<string>>dp(n+1,vector<string>());
         dp[0].push_back("");
    
          for(int i = 0; i < n; ++i){
            for(int j = i+1; j <= n; ++j){
                string temp = s.substr(i, j-i);
                if(word_Set.count(temp)){
                    for(auto x : dp[i]){
                        dp[j].emplace_back(x + (x == "" ? "" : " ") + temp);  
                    }
                }
            }
        }
         return dp[n];
    }


 CONCATENATED WORDS

 class Solution {
public:
    vector<string> findAllConcatenatedWordsInADict(vector<string>& words) {
        unordered_set<string>word_set(words.begin(),words.end());
        vector<string>ans;
        for(auto w:words){
            int n=w.size();
            vector<bool>dp(n+1,false);
            dp[0]=1;
            for(int i=0;i<n;i++){
                if(!dp[i])continue;
                for(int j=i+1;j<=n;j++){
                    if(j-i<n and word_set.count(w.substr(i,j-i)))
                    dp[j]=1;
                }
            }
            if(dp[n]==1)ans.push_back(w);
        }
       return ans;
    }
};


----------------------------------------------------------------------------------------------------------------

188. Best Time to Buy and Sell Stock IV -- at most k transaction --->
class Solution {
public:
    int helper(vector<int>& prices,vector<vector<vector<int>>>&dp,int index,int buy,int cnt){
        if(index==prices.size() || cnt==0)return 0;
        if(dp[index][buy][cnt]!=-1)return dp[index][buy][cnt];
        int profit = 0;
        if(buy==1){
            profit = max(-prices[index]+helper(prices,dp,index+1,0,cnt),helper(prices,dp,index+1,1,cnt));

        }
        if(buy==0){
            profit = max(prices[index]+helper(prices,dp,index+1,1,cnt-1),helper(prices,dp,index+1,0,cnt));
        }
        return dp[index][buy][cnt]=profit;
    }
    int maxProfit(int k, vector<int>& prices) {
        vector<vector<vector<int>>>dp(prices.size(),vector<vector<int>>(2,vector<int>(k+1,-1)));
        return helper(prices,dp,0,1,k);
        
        
    }
};

----------------------------------------------------------------------------------------------------------------------

198. HOUSE ROBBER --->  dp[i]=max(dp[i-1],dp[i-2]+nums[i-1]);
class Solution {
public:
    int rob(vector<int>& nums) {
        if(nums.size()==0)return 0;
        int n=nums.size();
        vector<int> dp(n+1,0);
        dp[1]=nums[0];
        for(int i=2;i<=n;i++){
            dp[i]=max(dp[i-1],dp[i-2]+nums[i-1]);
        }
        return dp[n];
    }
};


---------------------------------------------------------------------------------------------------------------------------------

213. House Robber II

class Solution {
public:
    int helper(int ind,vector<int>&v,vector<int>&dp){
        //if(ind==0)return v[ind];
        if(ind<0)return 0;
        if(dp[ind]!=-1)return dp[ind];
        return dp[ind]=max(v[ind]+helper(ind-2,v,dp),helper(ind-1,v,dp));
    }
    int rob(vector<int>& nums) {
        if(nums.size()==1)return nums[0];
        vector<int>dp1(nums.size(),-1),dp2(nums.size(),-1);
        vector<int>v1(nums.begin(),nums.end()-1);
        vector<int>v2(nums.begin()+1,nums.end());
        return max(helper(v1.size()-1,v1,dp1),helper(v2.size()-1,v2,dp2));
        
    }
};

----------------------------------------------------------------------------------------------------------------------------------

221. Maximal Square ---> Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.
     dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;

class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if (matrix.empty()) {
            return 0;
        }
        int m = matrix.size(), n = matrix[0].size(), sz = 0;
        vector<vector<int>> dp(m, vector<int>(n, 0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!i || !j || matrix[i][j] == '0') {
                    dp[i][j] = matrix[i][j] - '0';
                } else {
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
                sz = max(dp[i][j], sz);
            }
        }
        return sz * sz;
    }
};


-------------------------------------------------------------------------------------------------------------------------------------------

233. Number of Digit One

-----------------------------------------------------------------------------------------------------------------

241. Different Ways to Add Parentheses
     Given a string expression of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. You may return the answer in any order.
     The test cases are generated such that the output values fit in a 32-bit integer and the number of different results does not exceed 104

     class Solution {
    bool isOperator(char ch) {
        return (ch == '+' || ch == '-' || ch == '*');
    }

    vector<int> getDiffWays(int i, int j, vector<vector<vector<int>>>& dp, string& expression) {

        // Return cached result if already calculated
        if(!dp[i][j].empty()) {
            return dp[i][j];
        }
        
        // If length of the substring is 1 or 2
        // we encounter our base case i.e. a number found.
        int len = j - i + 1;
        if(len <= 2) {
            return dp[i][j] = { stoi(expression.substr(i, len)) };
        }

        // If it is not a number then it is an expression
        // now we try to evaluate every opertor present in it
        vector<int> res;
        for(int ind = i; ind <= j; ind++) {
            if(isOperator(expression[ind])) {
                char op = expression[ind];

                // if char at ind is operator 
                // get all results for its left and right substring using recursion
                vector<int> left = getDiffWays(i, ind - 1, dp, expression);
                vector<int> right = getDiffWays(ind + 1, j, dp, expression);

                // try all options for left & right operand
                // and push all results to the answer
                for(int l : left) {
                    for(int r : right) {
                        if(op == '+') {
                            res.push_back(l + r);
                        }
                        else if(op == '-') {
                            res.push_back(l - r);
                        }
                        else if(op == '*') {
                            res.push_back(l * r);
                        }
                    }
                }
            }
        }
        return dp[i][j] = res;
    }

public:
    vector<int> diffWaysToCompute(string expression) {
        int n = expression.size();
        vector<vector<vector<int>>> dp(n, vector<vector<int>>(n));
        return getDiffWays(0, n - 1, dp, expression);
    }
};


--------------------------------------------------------------------------------------------------------------------------

264. Ugly Number II
     An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5. Given an integer n, return the nth ugly number.

     struct Solution {
    int nthUglyNumber(int n) {
        vector <int> results (1,1);
        int i = 0, j = 0, k = 0;
        while (results.size() < n)
        {
            results.push_back(min(results[i] * 2, min(results[j] * 3, results[k] * 5)));
            if (results.back() == results[i] * 2) ++i;
            if (results.back() == results[j] * 3) ++j;
            if (results.back() == results[k] * 5) ++k;
        }
        return results.back();
    }
};

--------------------------------------------------------------------------------------------------------------------------------------

279. Perfect Squares
     Given an integer n, return the least number of perfect square numbers that sum to n.
     A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.
    
    class Solution {
public:
    int numSquares(int n) {
        //vector for updating the dp array/values
        vector<int> dp(n+1,INT_MAX);
        //base case
        dp[0]=0;
        int count = 1;
        while(count*count <= n) {
        int sq = count*count;
        for(int i = sq; i < n+1; i++) {
            dp[i] = min(dp[i-sq] + 1,dp[i]);
        }
        count++;
    }
    return dp[n];
    }
};


class Solution {
public:
    int helper(int i, int n, vector<int>&dp){
        if(n==0) return 0;
        if(dp[n]!=-1) return dp[n];
        if(i*i <= n){
            return dp[n]=min(1+helper(i, n-i*i, dp), helper(i+1, n, dp));
        }
        return dp[n]=1e5;
    }
    int numSquares(int n) {
        vector<int>dp(n+1, -1);
        return helper(1, n, dp);
    }
};


---------------------------------------------------------------------------------------------------------------------------------------------------

300. Longest Increasing Subsequence ----> 2D DP hota hai as the last taken value is important --> 1 9 6 7 8 ---> yaha 9 ko lenge ya nhi lenge(both cases ko consider krne ke lie we use 2d)

class Solution {
public:
    int solve(vector<int>& nums,int index,int prev_ind,vector<vector<int>>&dp){
        if(index==nums.size())return 0;
        if(dp[index][prev_ind+1]!=-1)return dp[index][prev_ind+1];
        int not_take = solve(nums,index+1,prev_ind,dp);      // bina check kre aage badh jao....
        int take=0;
        if(prev_ind==-1 || nums[index]>nums[prev_ind])take = 1 + solve(nums,index+1,index,dp);  // check karo aur jab include kr sko include kr lo
        return dp[index][prev_ind+1]= max(take,not_take);
    }
    int lengthOfLIS(vector<int>& nums) {
        vector<vector<int>>dp(nums.size(),vector<int>(nums.size()+1,-1));
        return solve(nums,0,-1,dp);
        
    }
};


----------------------------------------------------------------------------------------------------------------------------------------------------------
309. Best Time to Buy and Sell Stock with Cooldown ---> yaha 2d dp hi hai -->k transaction wale problem me we used 3d dp as har state pr k ka value was important pr yaha prham sifhe uske index me hi incorporate kr le rhe (cooldown ko so no need for3d dp)

class Solution {
public:
    int helper(vector<int>&prices,int index,int buy,vector<vector<int>>&dp){
        if(index>=prices.size())return 0;
        int max_p = 0;
        if(dp[index][buy]!=-1)return dp[index][buy];
        if(buy){
            max_p =  max(-prices[index]+helper(prices,index+1,0,dp),helper(prices,index+1,1,dp));
        }
        else{
          max_p =  max(prices[index]+helper(prices,index+2,1,dp),helper(prices,index+1,0,dp));
        }
        return dp[index][buy] =  max_p;


    }
    int maxProfit(vector<int>& prices) {
        vector<vector<int>>dp(prices.size(),vector<int>(2,-1));
        return helper(prices,0,1,dp);

        
    }
};

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

312. Burst Balloons -- You are given n balloons, indexed from 0 to n - 1. Each balloon is painted with a number on it represented by an array nums. You are asked to burst all the balloons.
     If you burst the ith balloon, you will get nums[i - 1] * nums[i] * nums[i + 1] coins. If i - 1 or i + 1 goes out of bounds of the array, then treat it as if there is a balloon with a 1 painted on it. Return the maximum coins you can collect by bursting the balloons wisely.

     class Solution {
public:
     
       int maxCoins(vector<int>& nums) {
        int c=nums.size();
        nums.insert(nums.begin(),1);
        nums.push_back(1);
        vector<vector<int>>dp(c+2,vector<int>(c+2,0));

        for(int i=c;i>=1;i--){
            for(int j=1;j<=c;j++){
                if(i>j)continue;
                int maxi=INT_MIN;
                for(int k=i;k<=j;k++){
                    maxi=max(maxi,nums[i-1]*nums[k]*nums[j+1]+dp[i][k-1]+dp[k+1][j]);
                }
                dp[i][j]=maxi;
            }
        }
        return dp[1][c];
        
    }
        
    
};

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

313. Super Ugly Number --> A super ugly number is a positive integer whose prime factors are in the array primes.
     Given an integer n and an array of integers primes, return the nth super ugly number.
     The nth super ugly number is guaranteed to fit in a 32-bit signed integer.

struct Solution {
    int nthUglyNumber(int n) {
        vector <int> results (1,1);
        int i = 0, j = 0, k = 0;
        while (results.size() < n)
        {
            results.push_back(min(results[i] * 2, min(results[j] * 3, results[k] * 5)));
            if (results.back() == results[i] * 2) ++i;
            if (results.back() == results[j] * 3) ++j;
            if (results.back() == results[k] * 5) ++k;
        }
        return results.back();
    }
};

---------------------------------------------------------------------------------------------------------------------

322. COIN CHANGE
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        vector<int>prev(amount+1,0);
        vector<int>curr(amount+1,0);

        
        for(int i=0;i<=amount;i++){
            if(i%coins[0]==0)prev[i]=i/coins[0];
            else prev[i]=1e9;
                    }
        for(int index = 1;index<coins.size();index++){
            for(int tot = 0;tot<=amount;tot++){
                int not_take = prev[tot];
                int take = 1e9;
                if(coins[index]<=tot)take = 1+curr[tot-coins[index]];
               curr[tot]= min(take,not_take);

            }
            prev=curr;
            
    
        }
        if(prev[amount]>=1e9)return -1;
        else return prev[amount];
        
    }
};

--------------------------------------------------------------------------------------------------------

class Solution {
public:


    int min_coins(vector<int>& coins, int amount,int index,vector<vector<int>>&dp){
        if(index==0){
            if(amount%coins[0]==0)return amount/coins[0];
            else return 1e9;
        }
        if(dp[index][amount]!=-1)return dp[index][amount];
        int not_take = min_coins(coins,amount,index-1,dp);
        int take = 1e9;
        if(coins[index]<=amount)take = 1+min_coins(coins,amount-coins[index],index,dp);
        return dp[index][amount]= min(take,not_take);
    }
    int coinChange(vector<int>& coins, int amount) {
        vector<vector<int>>dp(coins.size(),vector<int>(amount+1,-1));
        int ans =  min_coins(coins,amount,coins.size()-1,dp);

        if(ans>=1e9)return -1;
        else return ans;

        
    }
};


-----------------------------------------------------------------------------------------------------------------

329. Longest Increasing Path in a Matrix
     Given an m x n integers matrix, return the length of the longest increasing path in matrix.
     From each cell, you can either move in four directions: left, right, up, or down. You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).

class Solution {
public:
    
    bool checkLimits(int i, int j, int n, int m)return (i>=0 and i<n and j>=0 and j<m);
    
    int func(vector<vector<int>> &matrix, vector<vector<int>> &dp, int i, int j, int n, int m){
        
        if(!checkLimits(i, j, n, m)) return 0;
        
        if(dp[i][j]!=-1) return dp[i][j];
        
        int c1 = 0, c2 = 0, c3 = 0, c4 =0;
        
        if(checkLimits(i+1, j, n, m) and matrix[i+1][j]>matrix[i][j]){
            c1 = func(matrix, dp, i+1, j, n, m);
        }
        
        if(checkLimits(i-1, j, n, m) and matrix[i-1][j]>matrix[i][j]){
            c2 = func(matrix, dp, i-1, j, n, m);
        }
        
        if(checkLimits(i, j+1, n, m) and matrix[i][j+1]>matrix[i][j]){
            c3 = func(matrix, dp, i, j+1, n, m);
        }
        
        if(checkLimits(i, j-1, n, m) and matrix[i][j-1]>matrix[i][j]){
            c4 = func(matrix, dp, i, j-1, n, m);
        }
        
        dp[i][j] = 1 + max(c1, max(c2, max(c3, c4)));
        
        return dp[i][j];        
        
    }
    
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        
        int n = matrix.size(), m = matrix[0].size();
        int ans = 1;
        vector<vector<int>> dp(n, vector<int> (m, -1));
        
        for(int i = 0; i<n;i++){
            for(int j=0;j<m;j++){
                if(dp[i][j]==-1){    // YAHI PAR MEMOIZATION (DP) USE KR RHE....
                    ans = max(ans, func(matrix, dp, i, j, n, m));
                }
            }
        }
        return ans;
        
        
        
    }
};

----------------------------------------------------------------------------------------------------------------------------------

337. House Robber III
    The thief has found himself a new place for his thievery again. There is only one entrance to this area, called root.
    Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree. It will automatically contact the police if two directly-linked houses were broken into on the same night.
    Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.

At first glance, the problem exhibits the feature of "optimal substructure": if we want to rob maximum amount of money from current binary tree (rooted at root), we surely hope that we can do the same to its left and right subtrees.

So going along this line, let's define the function rob(root) which will return the maximum amount of money that we can rob for the binary tree rooted at root; the key now is to construct the solution to the original problem from solutions to its subproblems, i.e., how to get rob(root) from rob(root.left), rob(root.right), ... etc.

Apparently the analyses above suggest a recursive solution. And for recursion, it's always worthwhile figuring out the following two properties:

    Termination condition: when do we know the answer to rob(root) without any calculation? Of course when the tree is empty ---- we've got nothing to rob so the amount of money is zero.

    Recurrence relation: i.e., how to get rob(root) from rob(root.left), rob(root.right), ... etc. From the point of view of the tree root, there are only two scenarios at the end: root is robbed or is not. If it is, due to the constraint that "we cannot rob any two directly-linked houses", the next level of subtrees that are available would be the four "grandchild-subtrees" (root.left.left, root.left.right, root.right.left, root.right.right). However if root is not robbed, the next level of available subtrees would just be the two "child-subtrees" (root.left, root.right). We only need to choose the scenario which yields the larger amount of money.
// YAHA BFS KA ALTERNATE ROWS KA SUM CALCULATE NHI KARE KYUNKI --> AISA HO SKTA HAI KI WE WANT ROW 2 AND ROW 5 {row 3 and 4 having very small value compared to 2 and 5}

public int rob(TreeNode root) {
    if (root == null) return 0;
    
    int val = 0;
    
    if (root.left != null) {
        val += rob(root.left.left) + rob(root.left.right);
    }
    
    if (root.right != null) {
        val += rob(root.right.left) + rob(root.right.right);
    }
    
    return Math.max(val + root.val, rob(root.left) + rob(root.right));
}

In step I, we only considered the aspect of "optimal substructure", but think little about the possibilities of overlapping of the subproblems. For example, to obtain rob(root), we need rob(root.left), rob(root.right), rob(root.left.left), rob(root.left.right), rob(root.right.left), rob(root.right.right); but to get rob(root.left), we also need rob(root.left.left), rob(root.left.right), similarly for rob(root.right). The naive solution above computed these subproblems repeatedly, which resulted in bad time performance. Now if you recall the two conditions for dynamic programming (DP): "optimal substructure" + "overlapping of subproblems", we actually have a DP problem. A naive way to implement DP here is to use a hash map to record the results for visited subtrees.

public int rob(TreeNode root) {
    return robSub(root, new HashMap<>());
}

private int robSub(TreeNode root, Map<TreeNode, Integer> map) {
    if (root == null) return 0;
    if (map.containsKey(root)) return map.get(root);
    
    int val = 0;
    
    if (root.left != null) {
        val += robSub(root.left.left, map) + robSub(root.left.right, map);
    }
    
    if (root.right != null) {
        val += robSub(root.right.left, map) + robSub(root.right.right, map);
    }
    
    val = Math.max(val + root.val, robSub(root.left, map) + robSub(root.right, map));
    map.put(root, val);
    
    return val;
}

Step III -- Think one step back

In step I, we defined our problem as rob(root), which will yield the maximum amount of money that can be robbed of the binary tree rooted at root. This leads to the DP problem summarized in step II.

Now let's take one step back and ask why we have overlapping subproblems. If you trace all the way back to the beginning, you'll find the answer lies in the way how we have defined rob(root). As I mentioned, for each tree root, there are two scenarios: it is robbed or is not. rob(root) does not distinguish between these two cases, so "information is lost as the recursion goes deeper and deeper", which results in repeated subproblems.

If we were able to maintain the information about the two scenarios for each tree root, let's see how it plays out. Redefine rob(root) as a new function which will return an array of two elements, the first element of which denotes the maximum amount of money that can be robbed if root is not robbed, while the second element signifies the maximum amount of money robbed if it is robbed.

Let's relate rob(root) to rob(root.left) and rob(root.right)..., etc. For the 1st element of rob(root), we only need to sum up the larger elements of rob(root.left) and rob(root.right), respectively, since root is not robbed and we are free to rob its left and right subtrees. For the 2nd element of rob(root), however, we only need to add up the 1st elements of rob(root.left) and rob(root.right), respectively, plus the value robbed from root itself, since in this case it's guaranteed that we cannot rob the nodes of root.left and root.right.

As you can see, by keeping track of the information of both scenarios, we decoupled the subproblems and the solution essentially boiled down to a greedy one. Here is the program:

public int rob(TreeNode root) {
    int[] res = robSub(root);
    return Math.max(res[0], res[1]);
}

private int[] robSub(TreeNode root) {
    if (root == null) return new int[2];
    
    int[] left = robSub(root.left);
    int[] right = robSub(root.right);
    int[] res = new int[2];

    res[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
    res[1] = root.val + left[0] + right[0];
    
    return res;
}

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

338.   Counting Bits -->
       Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.
ans -> v.push_back(__builtin_popcount(i));

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

343. Integer Break
     Given an integer n, break it into the sum of k positive integers, where k >= 2, and maximize the product of those integers.
     Return the maximum product you can get.

    class Solution {
public:
    int integerBreak(int n) {
        // Base cases: for n < 4, return n-1
        if (n < 4)
            return n - 1;

        // Calculate the number of times 3 can be multiplied
        int numOfThrees = n / 3;

        // Initialize the answer with the product of 3 raised to numOfThrees
        long long productOfThrees = pow(3, numOfThrees);

        // Adjust the product based on the remainder when divided by 3
        if (n % 3 == 1) {
            // If remainder is 1, you can multiply by 4 instead of 3
            productOfThrees /= 3;
            productOfThrees *= 4;
        } else if (n % 3 == 2) {
            // If remainder is 2, multiply by 2
            productOfThrees *= 2;
        }

        // Return the maximum product
        return productOfThrees;
    }
};

**********
This question is a classic rod cutting problem a subset of unbounded knapsack problem.
The only catch is that we are multiplying with idx as we need to maximise the product
**********
class Solution {
public:
    int dp[59][58];
    
    int helper(int n, int idx)
    {
       if(n == 0 or idx == 0) return 1;
        
       if(dp[n][idx] != -1) return dp[n][idx];
        
       if(idx > n) return dp[n][idx] = helper(n, idx - 1);
      
       return dp[n][idx] = max((idx * helper(n - idx, idx)), helper(n , idx - 1));
    }
    
    int integerBreak(int n)
    {
        memset(dp, -1, sizeof dp);
        
        return helper(n, n - 1);
    }
};

-------------------------------------------------------------------------------------------------------------------

357. Count Numbers with Unique Digits
     Given an integer n, return the count of all numbers with unique digits, x, where 0 <= x < 10n.

    class Solution {
public:
    int countNumbersWithUniqueDigits(int n) {
        
        // n=0, return 1;
        // n=1, return 9 + dp[0]          =   9 + 1   = 10
        // n=2; return 9*9  +dp[1]        =  81 + 10  = 91
        // n=3, return 9*9*8 + dp[2]      = 648 + 91  = 739
        // n=4, return 9*9*8*7 + dp[3]    = 4536+ 739 = 5275
        //............
         
        vector<int> dp(n+1);
        dp[0]=1;
        
        int base =9, factor=9;
        for(int i=1; i<=n; i++){
            
            if( i>1)
                base*= (factor--); // 9*9 = 81 *8
            
            dp[i]= base + dp[i-1];
        }
        
        return dp[n];
    }
};


----------------------------------------------------------------------------------------------------

LIS: Once we have sorted our array, then this problem becomes very similar to LIS. It's just that instead of finding the Longest Increasing Subsequence, we want to look for the Longest Divisible Subsequence/Subset. Hence the algorithm will slightly vary, but the core idea will remain the same. So here, dp[i] will represent the length of the longest divisible subsequence which ends at the ith index. The state change would look something like this:
obtaining Subset: Now that we have the length of the Longest Divisible Subset and know the index where it ends, how do be obtain the entire set? From the above mentioned pseudo code, it is clear that for every dp[i], there will be one dp[j] which will preceed it. So we basically need to store this jth index which represents the predecessor of the ith index. For this, we can simply use another array which tracks predecessors of every index!

The code is very similar to the traditional LIS. I have used an extra array child to track predecessors of indices. Hence, dp[i] will tell the length of the Longest Divisible Subset ending at the ith index, and child[i] will tell the index which comes before the ith index in the subset which includes the ith index. child[i] will be -1, if the ith index is the first element in the subset.

class Solution {
public:
    vector<int> largestDivisibleSubset(vector<int>& nums) {
        
        // Sorting: to leverage the transitive property!
        sort(nums.begin(), nums.end());
        int n = nums.size();
        
        // Initializing the DP and CHILD arrays
        vector<int> dp(n, 1), child(n, -1);
        
        int imax = 0;
        
        for(int i = 1; i < n; i++) {
            
            // Considering ith element as the last element
            // and finding the largest subset it can 
            // belong to
            for(int j = 0; j < i; j++) {
                
                // Using transitivity, if nums[i] % nums[j] == 0
                // then all numbers in the subset ending at j
                // will divide num[i] as well!
                if(nums[i] % nums[j] == 0) {
                    
                    // Inclusion of i will increase the size of 
                    // the subset by 1
                    if(1 + dp[j] > dp[i]) {
                        dp[i] = 1 + dp[j];
                        
                        // Setting the predecessor of i
                        child[i] = j;
                    }
                }
            }
            
            // Determining the index where the largest subset ends
            if(dp[i] > dp[imax]) {
                imax = i;
            }
        }
        
        vector<int> ans;
        
        // Backtracking to obtain the entire largest subset!
        // This condition makes sure that the loop stops
        // once the first element of the subset is traversed
        while(imax != -1) {
            ans.push_back(nums[imax]);
            imax = child[imax];
        }
        
        return ans;
    }
};

------------------------------------------------------------------------------------

368.Largest Divisible Subset
    Given a set of distinct positive integers nums, return the largest subset answer such that every pair (answer[i], answer[j]) of elements in this subset satisfies:
    answer[i] % answer[j] == 0, or
    answer[j] % answer[i] == 0


class Solution {
public:
    vector<int> largestDivisibleSubset(vector<int>& nums) {
        
        // Sorting: to leverage the transitive property!
        sort(nums.begin(), nums.end());
        int n = nums.size();
        
        // Initializing the DP and CHILD arrays
        vector<int> dp(n, 1), child(n, -1);
        
        int imax = 0;
        
        for(int i = 1; i < n; i++) {
            
            // Considering ith element as the last element
            // and finding the largest subset it can 
            // belong to
            for(int j = 0; j < i; j++) {
                
                // Using transitivity, if nums[i] % nums[j] == 0
                // then all numbers in the subset ending at j
                // will divide num[i] as well!
                if(nums[i] % nums[j] == 0) {
                    
                    // Inclusion of i will increase the size of 
                    // the subset by 1
                    if(1 + dp[j] > dp[i]) {
                        dp[i] = 1 + dp[j];
                        
                        // Setting the predecessor of i
                        child[i] = j;
                    }
                }
            }
            
            // Determining the index where the largest subset ends
            if(dp[i] > dp[imax]) {
                imax = i;
            }
        }
        
        vector<int> ans;
        
        // Backtracking to obtain the entire largest subset!
        // This condition makes sure that the loop stops
        // once the first element of the subset is traversed
        while(imax != -1) {
            ans.push_back(nums[imax]);
            imax = child[imax];
        }
        
        return ans;
    }
};


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

377. Combination Sum IV
     Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.
     The test cases are generated so that the answer can fit in a 32-bit integer.

    class Solution {
public:
    int helper(vector<int>& nums, int target,int ind){
        if(ind<0){
            if(target==0)return 1;
            else return 0;

        }
        if(ind==0){
            if(target%nums[0]==0)return target/nums[0];
            else if(nums[0]==0 && target==0)return 2;
    
        }
        int l =0;
        if(target>=nums[ind])l= helper(nums,target-nums[ind],ind);
        int r = helper(nums,target,ind-1);
        return l+r;
    }
    int combinationSum4(vector<int>& nums, int target) {
        sort(nums.begin(),nums.end());
        int ind=nums.size()-1;
        return helper(nums,target,ind);
        
    }
};


If you accidentally remember the code in Coin Change 2, you may find the solution to this problem is basically the same with that, except the order of for loop.

class Solution { // Coin Change 2
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = 1; i <= amount; i++) {
                if (i >= coin) {
                    dp[i] += dp[i - coin];    
                }
            }
        }
        return dp[amount];
    }
}

class Solution { // this problem
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int n : nums) {
                if (i >= n) {
                    dp[i] += dp[i - n];
                }
            }
        }
        return dp[target];
    }
}

In this problem, we are required to count the duplicate results. However, in coin change, 1 + 1 + 2 is the same with 1 + 2 + 1. The order of the for loop actually makes these two different problems.


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
392. Is Subsequence --> Given two strings s and t, return true if s is a subsequence of t, or false otherwise.
/*class Solution {
public:
    void ans(string s, string t,int index,unordered_set<string>&st,string& tmp){
        if(index==t.length()){
            if(tmp.size()==s.length()){
                st.insert(tmp);
               
            }
             return;
        }
        
        tmp.push_back(t[index]);
        ans(s,t,index+1,st,tmp);
        tmp.pop_back();
        ans(s,t,index+1,st,tmp);
        

    }
    bool isSubsequence(string s, string t) {
        unordered_set<string>st;
        string tmp="";
        ans(s,t,0,st,tmp);
        if(st.find(s)!=st.end())return true;
        else return false;    }
};


**# Recursive Approach**

The idea is simple, we traverse both strings from one side to other side (say from rightmost character to leftmost). If we find a matching character, we move ahead in both strings. Otherwise we move ahead only in str2.

Code:
class Solution {
public:
    bool isSubSeq(string str1, string str2, int m, int n) 
{ 
    // Base Cases 
    if (m == 0) return true; 
    if (n == 0) return false; 
    // If last characters of two strings are matching 
    if (str1[m-1] == str2[n-1]) 
        return isSubSeq(str1, str2, m-1, n-1); 
    // If last characters are not matching 
    return isSubSeq(str1, str2, m, n-1); 
} 
    bool isSubsequence(string s, string t) {
        int m = s.size();
        int n = t.size();
        return isSubSeq(s,t,m,n);
    }
};



// Approach Using TWO POINTER

class Solution {
public:
    bool isSubsequence(string s, string t) {
        int m = s.size();
        int n = t.size();
        int i = 0, j = 0;
        while(i < m && j < n) {
            if(s[i] == t[j]) {
                i++;
            }
            j++;
        }
        return i == m ? 1 : 0;
    }
};


//  USING DYNAMIC PROGRAMMING 
// if LCS of string A  nd B is equal to Size of String A then its True else false;

class Solution {
public:
    int helper(string x, string y) {
        int m = x.size();
        int n = y.size();
        int dp[m+1][n+1];
        for(int i = 0; i<=m; i++)
        {
            dp[i][0]=0;
        }
        for(int i = 0; i<=n; i++)
        {
            dp[0][i]=0;
        }
        for(int i = 1; i<=m; i++){
            for(int j = 1; j<=n; j++){
                if(x[i-1]==y[j-1]){
                    dp[i][j]=1+dp[i-1][j-1];
                }else{
                    dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[m][n];
    }
    bool isSubsequence(string smaller, string larger) {

        int x  = helper(smaller,larger);
        if(x == smaller.size()){
            return true;
        }
        return false;      
    }
};


--------------------------------------------------------------------------------------------------------------------------------------------------------------------
396. Rotate Function
     You are given an integer array nums of length n. Assume arrk to be an array obtained by rotating nums by k positions clock-wise. We define the rotation function F on nums as follow:
     F(k) = 0 * arrk[0] + 1 * arrk[1] + ... + (n - 1) * arrk[n - 1].
     Return the maximum value of F(0), F(1), ..., F(n-1). The test cases are generated so that the answer fits in a 32-bit integer.
 
    // F(k) = F(k-1) + sum - nBk[0]
// k = 0; B[0] = A[0];
// k = 1; B[0] = A[len-1];
// k = 2; B[0] = A[len-2];

class Solution {
public:
    int maxRotateFunction(vector<int>& nums) {
        int sum =0;
        int f=0;
        for(int i=0;i<nums.size();i++){
			sum+=nums[i];
			f+=i*nums[i];
		}
        
        int globalSum = f;
        
        for(int i=nums.size()-1;i>0;i--){
            f = f + sum -nums.size()*nums[i];
            globalSum = max(f,globalSum);
        }
        return globalSum;
    }
};

-------------------------------------------------------------------------------------------------------------------------------------------------

397. Integer Replacement

Given a positive integer n, you can apply one of the following operations:
If n is even, replace n with n / 2.
If n is odd, replace n with either n + 1 or n - 1.
Return the minimum number of operations needed for n to become 1.

class Solution {
public:

long long integerReplace(long long n, unordered_map<long long, int> &dp){
    if(n<=1)
    return 0;

    if(dp.find(n)!=dp.end())
    return dp[n];
        
    if(n%2==0)return dp[n]=1+integerReplace(n/2,dp);
    
    else return dp[n]= 1+min(integerReplace(n+1,dp),integerReplace(n-1,dp));
}
    int integerReplacement(int n) {
        unordered_map<long long,int> dp;
        return integerReplace(n, dp);
        
    }
};

-----------------------------------------------------------------------------------------------------------------------------------
403. FROG JUMP
A frog is crossing a river. The river is divided into some number of units, and at each unit, there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.
Given a list of stones positions (in units) in sorted ascending order, determine if the frog can cross the river by landing on the last stone. Initially, the frog is on the first stone and assumes the first jump must be 1 unit.
If the frog's last jump was k units, its next jump must be either k - 1, k, or k + 1 units. The frog can only jump in the forward direction.

class Solution {
public:
    map<int,int> mp;
    bool canCross(vector<int>& st) {
        int n = st.size();
        vector<vector<int>> dp(n,vector<int> (2*n,-1));
        for(int i = 1 ; i < n ; i++){
            mp[st[i]] = i ;
        }
        if(st[1] - st[0] > 1){
            return false;
        }
        return f(1,1,st,dp);
    }
    bool f(int i , int k, vector<int>& st,vector<vector<int>>&dp){
        if(i == st.size() - 1){
            return true;
        }
        bool bk = false;
        bool atk = false;
        bool ak = false;
        if(dp[i][k] !=-1){
            return dp[i][k];
        }
        //<!-- For k - 1 -->
        if(mp[st[i] + k-1] > i){
            bk = f(mp[st[i] + k-1],k-1,st,dp);
        }
       // <!-- For k -->
        if(mp[st[i] + k] > i){
            atk = f(mp[st[i] + k],k,st,dp);
        }
        //<!-- For k + 1 -->
        if(mp[st[i] + k+1] > i){
            ak = f(mp[st[i] + k+1],k+1,st,dp);
        }
        return dp[i][k] = (bk || atk || ak);
    }
};

------------------------------------------------------------------------------------------------------------------------------

410. Split Array Largest Sum
Given an integer array nums and an integer k, split nums into k non-empty subarrays such that the largest sum of any subarray is minimized.
Return the minimized largest sum of the split. A subarray is a contiguous part of the array.

class Solution {
public:

    int countStudents(vector<int> &arr, int pages) {
    int n = arr.size(); //size of array.
    int students = 1;
    long long pagesStudent = 0;
    for (int i = 0; i < n; i++) {
        if (pagesStudent + arr[i] <= pages) {
            //add pages to current student
            pagesStudent += arr[i];
        }
        else {
            //add pages to next student
            students++;
            pagesStudent = arr[i];
        }
    }
    return students;
}

int findPages(vector<int>& arr, int n, int m) {
    //book allocation impossible:
    if (m > n) return -1;

    int low = *max_element(arr.begin(), arr.end());
    int high = accumulate(arr.begin(), arr.end(), 0);

    while(low<=high){
        int mid = (low+high)/2;
        if(countStudents(arr,mid)>m){
            low=mid+1;

        }
        else high=mid-1;
    }
    return low;
}
    int splitArray(vector<int>& nums, int k) {
        return findPages(nums,nums.size(),k);
        
    }
};

----------------------------------------------------------------------------------------------------------------------------

413. Arithmetic Slices -->An integer array is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.
     For example, [1,3,5,7,9], [7,7,7,7], and [3,-1,-5,-9] are arithmetic sequences.
     Given an integer array nums, return the number of arithmetic subarrays of nums.
     A subarray is a contiguous subsequence of the array.
     
    //Solution 01:
//Recursion
class Solution {
public:
    int sum = 0;
    int numberOfArithmeticSlices(vector<int>& A) {
        
                slices(A, A.size() - 1);
        return sum;

        
    }
    
    
     int slices(vector<int>& A, int i) {
        if (i < 2)
            return 0;
        int ap = 0;
        if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]) {
            ap = 1 + slices(A, i - 1);
            sum += ap;
        } else
            slices(A, i - 1);
        return ap;
    }
};

//DP

class Solution {
public:
    int sum = 0;
    int numberOfArithmeticSlices(vector<int>& A) {
        
   vector<int> dp(A.size());
        int sum = 0;
        for (int i = 2; i < dp.size(); i++) {
            if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]) {
                dp[i] = 1 + dp[i - 1];
                sum += dp[i];
            }
        }
        return sum;
     
    }
    
    
    
};


//DP with constant space

class Solution {
public:
    int sum = 0;
    int numberOfArithmeticSlices(vector<int>& A) {
        
        int sum = 0,cur = 0;
        for (int i = 2; i < A.size(); i++) {
            if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]) {
                cur += 1;
                sum += cur;
            }
            else
                cur = 0;
        }
        return sum;
     
    }
    
    
    
};

//Using Formula

class Solution {
public:
    int sum = 0;
    int numberOfArithmeticSlices(vector<int>& A) {
        
               int count = 0;
        int sum = 0;
        for (int i = 2; i < A.size(); i++) {
            if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]) {
                count++;
            } else {
                sum += (count + 1) * (count) / 2;
                count = 0;
            }
        }
        return sum += count * (count + 1) / 2;
    }
    
    
    
};

----------------------------------------------------------------------------------------------------------------------

416. Partition Equal Subset Sum
     Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.
---> sum of subsequence == (total_sum)/2

class Solution {
public:
    int dp[201][20001];
    bool solve(vector<int> &nums, int n, int sum)
    {
        if (n <= 0 || sum <= 0)
            return sum == 0;
        
        if (dp[n][sum] != -1)
            return dp[n][sum];
        
        if (nums[n-1] > sum)
            return dp[n][sum] = solve(nums, n-1, sum);
        else
            return dp[n][sum] = solve(nums, n-1, sum) || solve(nums, n-1, sum-nums[n-1]);
    }
    
    bool canPartition(vector<int>& nums) 
    {
        int sum = 0;
        memset(dp, -1, sizeof(dp));
        
        for(int i = 0; i < nums.size(); i++)
            sum += nums[i];
        
        if (sum % 2 != 0) 
            return false;
        
        return solve(nums, nums.size(), sum/2);
    }
};


--------------------------------------------------------------------------------------------------------------------------------------------------

class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for (auto a : nums) // Sum up the array
            sum += a;
        
        if (sum % 2) // If the sum is odd - we can never find two equal partitions
            return false;
        
        sum /= 2;
        vector<bool> dp(sum+1, false); // dp keeps for each number if it has a subset or not
        dp[0] = true;
        
        for (auto a : nums) {
            for (int i = sum; i >= a; i--) {
                dp[i] = dp[i] || dp[i-a]; // for each number, either we use it or we don't
            } 
        }
        return dp[sum];
    }
};

class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for(int i =0;i<nums.size();i++){
            sum+= nums[i];
        }
        if(sum%2) return false;
        sum /= 2;
        sort(nums.rbegin(),nums.rend());
        return helper(nums, sum, 0);
    }
    bool helper(vector<int>& nums, int sum, int index){
        if(sum == nums[index]) return true;
        if(sum < nums[index]) return false;
        return helper(nums,sum-nums[index],index+1) || helper(nums,sum,index+1);
    }
};


------------------------------------------------------------------------------------------------------------------------------------------------------------------------

435. Non-overlapping Intervals

we have to sort the intervals on the basis of thier end points,
then use a greeady approach to find the answer.

If p is ending after the start of current element, we eliminate the current element but not the element contained in p because the elements are sorted according to their end points and p will have a lesser end point than the current element. So we eliminate current element to reduce the probability of overlapping with next element.

bool comp(vector<int> a, vector<int> b)
{
    if(a[1] == b[1]) return a[0]<b[0];
    return a[1]<b[1];
}

int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(),comp);
        int n = intervals.size();
        int ans = 0;
        vector<int> p = intervals[0];
        
        for(int i=1;i<n;i++)
        {
            if(p[1] > intervals[i][0])
                ans++;
            else
                p = intervals[i];
        }
        return ans;
    }

Why do we sort on the basis of end points, not on start points.

    suppose you have cases like : (1,8), (2,3), (3,4), (5,9)
    if you sort in the basis of start points you will end up considering (1,8) and deleting rest which collide with (1,8).
    For a greedy approach you will want the point with lower end point to be considered.
    But, We can sort on the basis of start point also, just a little tweak in the algorithm will work out that way. In case of overlap, remove the interval having the farther end point.


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

473. Matchsticks to Square

class Solution {
	// a,b,c,d are four sides of square
    int a,b,c,d;
    bool fun(vector<int>& matchsticks,int i){
        //Base Case
        if(i==matchsticks.size()){
            if(a==0 && b==0 && c==0 && d==0) return true;
            else return false;
        }
        
		//Now we will explore for all side for given index
		
		// if matchstick size is less than side(a or b or c or d)  size , then in that case we will not explore that because that will cause negative side which is not possible
        if(matchsticks[i]<=a){
            a-=matchsticks[i];
            if(fun(matchsticks,i+1)) return true;
            a+=matchsticks[i];      // backtrack step
        }
        
        if(matchsticks[i]<=b){
            b-=matchsticks[i];
            if(fun(matchsticks,i+1)) return true;
            b+=matchsticks[i];        // backtrack step                    
        }
        
        if(matchsticks[i]<=c){
            c-=matchsticks[i];
            if(fun(matchsticks,i+1)) return true;
            c+=matchsticks[i];         // backtrack step
        }
        
        if(matchsticks[i]<=d){
            d-=matchsticks[i];
            if(fun(matchsticks,i+1)) return true;
            d+=matchsticks[i];         // backtrack step
        }
		
		//If none of the explored option retuen true then  we have to return false
        return false;
    }
public:
    bool makesquare(vector<int>& matchsticks) {
		//  if less than four number present in array , then we can not make square
        if(matchsticks.size()<4) return false;
        
		// if sum of all number of array is not divisible by 4 , then we can not create a square
		int sum = accumulate(matchsticks.begin(), matchsticks.end(),0);
        if(sum % 4 != 0) return false;
        
		int sizeSum=sum/4;
        a=sizeSum,b=sizeSum,c=sizeSum,d=sizeSum;
        
		// here we sort our array in reverse order to escape more cases
		sort(matchsticks.rbegin(), matchsticks.rend());
        
		return fun(matchsticks,0);
    }
};


---------------------------------------------------------------------------------------------------------------------------------------------------------------'

474. Ones and Zeroes

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
486. Predict the Winner
     You are given an integer array nums. Two players are playing a game with this array: player 1 and player 2.
     Player 1 and player 2 take turns, with player 1 starting first. Both players start the game with a score of 0. At each turn, the player takes one of the numbers from either end of the array (i.e., nums[0] or nums[nums.length - 1]) which reduces the size of the array by 1. The player adds the chosen number to their score. The game ends when there are no more elements in the array.
     Return true if Player 1 can win the game. If the scores of both players are equal, then player 1 is still the winner, and you should also return true. You may assume that both players are playing optimally.
---> think why we need DP -->
     GREEDY --> pick the greater of the two elements at the end points..to maximise the score of the 1st(current) player
     GREEDY DOESN'T WORK HERE --> as agar 7 11 9 5 raha then picking 6 gives oppenent {7 ans 9}
                                                         ans picking 7 gives opponent {10 and 6}
    two case -->picking form start / picking from end --- and use dp
---> 2 player hai and do direction hai so obviously 2d dp   

class Solution {
public:

    int helper(int a,vector<int>& nums,int i,int j,vector<vector<vector<int>>>&dp){
        if(i>j)return 0;
        if(dp[i][j][a]!=-1)return dp[i][j][a];
        int ans;
        if(a)ans=INT_MIN;
        else ans=INT_MAX;
        if(a)ans = max(nums[i]+helper(!a,nums,i+1,j,dp),nums[j]+helper(!a,nums,i,j-1,dp));
        else ans=min(helper(!a,nums,i+1,j,dp),helper(!a,nums,i,j-1,dp));
        return dp[i][j][a]=ans;
    }
    bool predictTheWinner(vector<int>& nums) {
        int sum=0;
        vector<vector<vector<int>>>dp(21,vector<vector<int>>(21,vector<int>(2,-1)));
        for(auto i:nums)sum+=i;
        int ans=helper(1,nums,0,nums.size()-1,dp);
        if(2*ans>=sum)return true;
        else return false;
        
    }
};

*

class Solution {
    public boolean PredictTheWinner(int[] nums) {
        
        int scoreFirst = predictTheWinnerFrom(nums, 0, nums.length - 1);
        int scoreTotal = getTotalScores(nums);
        
        // Compare final scores of two players.
        return scoreFirst >= scoreTotal - scoreFirst;
    }
    
    private int predictTheWinnerFrom(int[] nums, int i, int j) {
        if (i > j) {
            return 0;
        }
        if (i == j) {
            return nums[i];
        }
        
        int curScore = Math.max(
            nums[i] + Math.min(
                predictTheWinnerFrom(nums, i + 2, j), 
                predictTheWinnerFrom(nums, i + 1, j - 1)
            ),
            nums[j] + Math.min(
                predictTheWinnerFrom(nums, i, j - 2), 
                predictTheWinnerFrom(nums, i + 1, j - 1)
            )
        );   
        return curScore;
    }
    
    private int getTotalScores (int[] nums) {
        int scoreTotal = 0;
        for (int num : nums) {
            scoreTotal += num;
        }
        
        return scoreTotal;
    }
}


----------------------------------------------------------------------------------------------------------------------------

494. Target Sum ---> ### PICK - NON-PICK SE SOLVE KRE HAI -- ISKO BHI .. PICKING ELEMENTS WHICH WE WANT TO BE POSITIVE
     You are given an integer array nums and an integer target.
     You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.
     For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
     Return the number of different expressions that you can build, which evaluates to target.
    
    class Solution {
public:
    int ans(vector<int>& nums, int target_1,int index,vector<vector<int>>&dp){
        if(index==0){
            if(target_1==0 && nums[0]==0)return 2;
            if(target_1==0 || target_1==nums[0])return 1;
            return 0;
        }
        if(dp[index][target_1]!=-1)return dp[index][target_1];
        int not_take = ans(nums,target_1,index-1,dp);
        int take=0;
        if(target_1-nums[index]>=0)take = ans(nums,target_1-nums[index],index-1,dp);
        return dp[index][target_1] = take+not_take;
    }
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum=0;
        int n=nums.size();
       
        for(int i=0;i<nums.size();i++)sum+=nums[i];
        if((sum+target)%2==1)return 0;
        int target_1 = (sum+target)/2;
         vector<vector<int>>dp(n,vector<int>(target_1+1,-1));
        return ans(nums,target_1,n-1,dp);

    }
};


class Solution {
public:
    int target;
    int knapsack(vector<int>& nums, int n, int curSum) {
        if(n == 0) {
            if(curSum == target) 
                return 1;
            return 0;
        }
        return knapsack(nums, n - 1, curSum + nums[n - 1]) + knapsack(nums, n - 1, curSum - nums[n - 1]);
        
    }
    
    int findTargetSumWays(vector<int>& nums, int sum) {
        target = sum;
        return knapsack(nums, nums.size(), 0);
    }
};

--------------------------------------------------------------------------------------------------------------------------------------------------
516. Longest Palindromic Subsequence --- Given a string s, find the longest palindromic subsequence's length in s.

class Solution {
public:
    int LCS(string s,string t){
        vector<vector<int>>dp(s.length()+1,vector<int>(t.length()+1,0));
        for(int index1=1;index1<=s.length();index1++){
            for(int index2=1;index2<=t.length();index2++){
                if(s[index1-1]==t[index2-1]){
                    dp[index1][index2]=1+dp[index1-1][index2-1];
                }else{
                    dp[index1][index2]=max(dp[index1][index2-1],dp[index1-1][index2]);
                }
            }
        }
        return dp[s.length()][t.length()];
    }
    int longestPalindromeSubseq(string s) {
        string t=s;
        reverse(s.begin(),s.end());
        int ans=LCS(s,t);
        return ans;
    }
};




---------------------------------------------------------------------------------------------------------------------------

518. Coin Change II
     You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
     Return the number of combinations that make up that amount. If that amount of money cannot be made up by any combination of the coins, return 0.
     You may assume that you have an infinite number of each kind of coin.

     class Solution {
public:
    int helper(int amount, vector<int>& coins,int index,vector<vector<int>>&dp){
        if(index==0)return (amount%coins[0]==0);
        if(dp[index][amount]!=-1)return dp[index][amount];

        int not_take = helper(amount,coins,index-1,dp);
        int take = 0;
        if(coins[index]<=amount)take = helper(amount-coins[index],coins,index,dp);
        return dp[index][amount]=take+not_take;
    }
    int change(int amount, vector<int>& coins) {
        vector<vector<int>>dp(coins.size(),vector<int>(amount+1,-1));
        return helper(amount,coins,coins.size()-1,dp);
        
    }
};

class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<vector<int>>dp(coins.size(),vector<int>(amount+1,0));
        for(int i=0;i<=amount;i++){
            if(i%coins[0]==0)dp[0][i]=1;
        }
        for(int index = 1;index<coins.size();index++){
            for(int tot = 0;tot<=amount;tot++){
                int not_take = dp[index-1][tot];
                int take = 0;
                if(coins[index]<=tot)take =dp[index][tot-coins[index]];
                 dp[index][tot]=take+not_take;

            }
        }
        
        return dp[coins.size()-1][amount];
        
    }
};




class Solution {
public:

    int change(int amount, vector<int>& coins) {
        vector<vector<int>>dp(coins.size(),vector<int>(amount+1,0));
        vector<int>prev(amount+1,0);
        vector<int>curr(amount+1,0);

        for(int i=0;i<=amount;i++){
            if(i%coins[0]==0)prev[i]=1;
        }
        for(int index = 1;index<coins.size();index++){
            for(int tot = 0;tot<=amount;tot++){
                int not_take = prev[tot];
                int take = 0;
                if(coins[index]<=tot)take =curr[tot-coins[index]];
                curr[tot]=take+not_take;

            }
            prev = curr;
        }
        
        return prev[amount];
        
    }
};

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
542. 01 Matrix
     Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.
     The distance between two adjacent cells is 1.


class Solution {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        int n=mat.size();
        int m=mat[0].size();

        queue<pair<pair<int,int>,int>>q;
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(mat[i][j]==0){
                    q.push({{i,j},0});
                    mat[i][j]=-1;

                }
            }
        }
        int dr[]={1,0,-1,0};
        int dc[]={0,1,0,-1};
        vector<vector<int>>answer(n,vector<int>(m,0));
        while(!q.empty()){
             int size=q.size();
             while(size--){
                 auto x=q.front();
                 q.pop();
                 int r=x.first.first;
                 int c=x.first.second;
                 int d=x.second;
                 for(int i=0;i<4;i++){
                     int new_r=r+dr[i];
                     int new_c=c+dc[i];
                     if(new_c>=0 && new_c<m && new_r>=0 && new_r<n){
                         if(mat[new_r][new_c]==1){
                             answer[new_r][new_c]=d+1;
                             q.push({{new_r,new_c},d+1});
                             mat[new_r][new_c]=-1;
                     }
                                                      
                }
                         

                     }
                 }
             }
             return answer;
    }

};


-----------------------------------------------------------------------------------------------------------------------
546. Remove Boxes

You are given several boxes with different colors represented by different positive numbers.
You may experience several rounds to remove boxes until there is no box left. Each time you can choose some continuous boxes with the same color (i.e., composed of k boxes, k >= 1), remove them and get k * k points.
Return the maximum points you can get.


---------------------------------------------------------------------------------------------------------------------------------------------

583. Delete Operation for Two Strings
     Given two strings word1 and word2, return the minimum number of steps required to make word1 and word2 the same.
     In one step, you can delete exactly one character in either string.
######  ISSE OBVIOUS , SIMPLE AND DIRECT IMPLEMENTATION NHI MILEGA LCS PATTERN KA
   
    class Solution {
public:
    
// Function to calculate the length of the Longest Common Subsequence
int lcs(string s1, string s2) {
    int n = s1.size();
    int m = s2.size();

    vector<vector<int>> dp(n + 1, vector<int>(m + 1, -1));

    // Initialize the first row and first column to 0
    for (int i = 0; i <= n; i++) {
        dp[i][0] = 0;
    }
    for (int i = 0; i <= m; i++) {
        dp[0][i] = 0;
    }

    for (int ind1 = 1; ind1 <= n; ind1++) {
        for (int ind2 = 1; ind2 <= m; ind2++) {
            if (s1[ind1 - 1] == s2[ind2 - 1])
                dp[ind1][ind2] = 1 + dp[ind1 - 1][ind2 - 1];
            else
                dp[ind1][ind2] = max(dp[ind1 - 1][ind2], dp[ind1][ind2 - 1]);
        }
    }

    return dp[n][m];
}
    int minDistance(string word1, string word2) {
        int k=lcs(word1,word2);
        int n=word1.length();
        int m=word2.length();
        return m-k+n-k;
        
    }
};

class Solution {
public:
    int helper(string word1,string word2,int i,int j,vector<vector<int>>&dp){
        if(i<0)return j+1;
        else if(j<0)return i+1;
        int no_del=INT_MAX,del=INT_MAX;
        if(word1[i]==word2[j]){
            no_del=helper(word1,word2,i-1,j-1,dp);
        }else{
            del=1+min(helper(word1,word2,i-1,j,dp),helper(word1,word2,i,j-1,dp));
        }
        return dp[i][j]= min(del,no_del);
    }
    int minDistance(string word1, string word2) {
        vector<vector<int>>dp(word1.length(),vector<int>(word2.length(),-1));
        return helper(word1,word2,word1.size()-1,word2.size()-1,dp);
    }
};

--------------------------------------------------------------------------------------------------------------------------------

646. Maximum Length of Pair Chain
You are given an array of n pairs pairs where pairs[i] = [lefti, righti] and lefti < righti.
A pair p2 = [c, d] follows a pair p1 = [a, b] if b < c. A chain of pairs can be formed in this fashion.
Return the length longest chain which can be formed. You do not need to use up all the given intervals. You can select pairs in any order.



/*
This question is a bit challenging to code. There is mainly to approaches to solve this:

I was able to come up with DP after reading the problem
Approach 1: Dynamic Programming -> O(n2)O(n^2)O(n2)
(It is intuitive approach)

    Sorting: Start by sorting the pairs based on their second element in ascending order. This sorting step ensures that we can build a chain with increasing second elements.

    Dynamic Programming: Utilize dynamic programming to keep track of the maximum chain length ending at each pair. Initialize a DP array with all values as 1, representing the minimum chain length (each pair can be a chain of length 1). This step helps us build upon valid chains iteratively.

    Updating DP Array: Iterate through the sorted pairs. For each pair, iterate through previous pairs. If the current pair's first element is greater than the previous pair's second element (i.e., a valid link), update the DP value for the current pair to be the maximum of its current value and the DP value of the previous pair plus one.

    Finding Maximum Length: After updating the DP array, the maximum chain length will be the maximum value within the DP array.

    Return Result: Return the maximum chain length as the final answer.

I was not able to come up with the greedy approach.
Approach 2: Greedy -> O(nlogn)O(nlogn)O(nlogn)
(Based on few observation you might come up with this approach or if you have past experience of such problems)

    Sorting: Start by sorting the pairs based on their second element in ascending order. This sorting step remains consistent with the dynamic programming approach.

    Greedy Strategy: Iterate through the sorted pairs while keeping track of the last ending element encountered. If the current pair's starting element is greater than the last ending element, it can be added to the chain, and the chain length is incremented.

    Finding Maximum Length: The maximum chain length will be the length of the chain obtained from the greedy approach.

    Result: Return the maximum chain length as the final answer.

If you have any other approach please let me know. I would be really grateful to learn that.
*/
class Solution {
private:
    // get the nearest pair after i, say (c, d), for which c > pairs[i][1]
    int binarySearch(vector<vector<int>>& pairs, int i) {
        int n = pairs.size();
        int start = i+1, end = n-1, ans = n;
        while(start <= end) {
            int mid = start + (end - start)/2;
            if(pairs[mid][0] > pairs[i][1]) {
                ans = mid;
                end = mid - 1;
            }
            else {
                start = mid + 1;
            }
        }
        return ans;
    }
public:
    int findLongestChain(vector<vector<int>>& pairs) {
        sort(pairs.begin(), pairs.end());
        int n = pairs.size();
        vector<int> dp(n+1, 0);
        // for the last element, we should choose it in any case
        dp[n-1] = 1;
        for(int i = n-2; i >= 0; --i) {
            int exclude = dp[i+1];
            int nextIdx = binarySearch(pairs, i);
            int include = 1 + dp[nextIdx];
            dp[i] = max(include, exclude);
        }
        return dp[0];
    }
};


class Solution {
public:
    int findLongestChain(vector<vector<int>>& pairs) {
        sort(pairs.begin(), pairs.end(), 
        [](vector<int>& p0, vector<int>& p1){
            return p0[1]<p1[1];
        });
    //    for(auto p: pairs) cout<<"("<<p[0]<<","<<p[1]<<")";
     
        int ans=0;
        int prev_r=INT_MIN;
        for (auto& p: pairs){
            if (p[0]> prev_r) {
                ans++;
                prev_r=p[1];
            }     
        }
        return ans;
    }
};

class Solution {
public:
    int findLongestChain(vector<vector<int>>& pairs) {
        sort(pairs.begin(), pairs.end());
    //    for(auto p: pairs) cout<<"("<<p[0]<<","<<p[1]<<")";
        int n=pairs.size();
        int ans=0;
        int prev_l=INT_MAX;
        for (int i=n-1; i>=0; i--){
            if (pairs[i][1]< prev_l) {
                ans++;
                prev_l=pairs[i][0];
            }     
        }
        return ans;
    }
};

-----------------------------------------------------------------------------------------------------------------------------------

647. Palindromic Substrings
     Given a string s, return the number of palindromic substrings in it.
     A string is a palindrome when it reads the same backward as forward.
     A substring is a contiguous sequence of characters within the string.
    len(S) < 1000

class Solution {
public:
    int countSubstrings(string s) {
       int cnt=0;
       for(int i=0;i<s.length();i++){
            int l=i,r=i;
            while(l>=0 && r<s.length() && s[l]==s[r]){
                cnt++;
                l--;
                r++;
        }
            l=i,r=i+1;
            while(l>=0 && r<s.length() && s[l]==s[r]){
                cnt++;
                l--;
                r++;
        }
       }
       return cnt;
    }
};

-------------------------------------------------------------------------------------------------------------------------------

650. 2 Keys Keyboard
There is only one character 'A' on the screen of a notepad. You can perform one of two operations on this notepad for each step:
Copy All: You can copy all the characters present on the screen (a partial copy is not allowed).
Paste: You can paste the characters which are copied last time.
Given an integer n, return the minimum number of operations to get the character 'A' exactly n times on the screen.


// return min step to reach the target value
// there are 2 choices: paste copied value with current value => 1 step
// copy current value and paste it with itself => 2 step

class Solution {
public:
        
    // dp vector to store <step,value> result for using in future
    int dp[1001][1001];
    
    int minKeyPress(int step, int value, int copy, int&n)
    {
        // impossible case when step>n or value>n, so return INT_MAX
        if(step>n || value>n) return INT_MAX;
        if(value==n) return step;
        if(dp[step][value]!=-1) return dp[step][value];
        return dp[step][value] = min(minKeyPress(step+1,value+copy,copy,n),minKeyPress(step+2,2*value,value,n));
    }

    int minSteps(int n){
        if(n==1) return 0;
        memset(dp,-1,sizeof(dp));
        
        // start with value 1 and copy 1 and intial step 1 (assuming we already copied the intial value 'A')
        return minKeyPress(1,1,1,n);
    }
};



We want to find the greatest divisor for our number, so we can minimize the number of steps by getting it in a buffer and 
pasting multiple times. The quickest way to find the greatest divisor is to start with the smallest prime and work our way up. 
Note that we only need primes up to 31 as n is limited to 1,000 (32 * 32 > 1,000).

When we find a prime i, the greatest divisor will be n / i. Then, we will recursively calculate the minimum steps for that greatest divisor.

class Solution {
public:
int minSteps(int n) {
    static const int primes[11] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31 };
    if (n <= 5) return n == 1 ? 0 : n;
    for (auto i : primes)
        if (n % i == 0) return i + minSteps(n / i);
    return n; // prime number.
}
};


f(n) = n, n is a prime number.


--------------------------------------------------------------------------------------------------------------------------------------------------

673. Number of Longest Increasing Subsequence
Given an integer array nums, return the number of longest increasing subsequences.
Notice that the sequence has to be strictly increasing.

class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n=nums.size();
        vector<int>dp(n,1);
        vector<int>ct(n,1);
        int maxi=-1;
        for(int i=0;i<n;i++){
            for(int pre_ind = 0;pre_ind<i;pre_ind++){
                if(dp[i]< 1 + dp[pre_ind] && nums[i]>nums[pre_ind]){
                    dp[i]=dp[pre_ind]+1;
                    ct[i]=ct[pre_ind];
                    
                }else if(dp[i]== 1 + dp[pre_ind] && nums[i]>nums[pre_ind]){
                    ct[i]+=ct[pre_ind];
                }
            }
            maxi=max(maxi,dp[i]);
        }

        int ans = 0;
        for(int i =0;i<n;i++){
            if(dp[i]==maxi){
                ans+=ct[i];            
            }
        }
        
        return ans;
    }
};

----------------------------------------------------------------------------------------------------------------------------------
678. Valid Parenthesis String


Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.

The following rules define a valid string:

    Any left parenthesis '(' must have a corresponding right parenthesis ')'.
    Any right parenthesis ')' must have a corresponding left parenthesis '('.
    Left parenthesis '(' must go before the corresponding right parenthesis ')'.
    '*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".

class Solution {
    public boolean checkValidString(String s) {
        int openCount = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                openCount++;
            } else if (c == ')') {
                openCount--;
            }
            if (openCount < 0) return false;    // Currently, don't have enough open parentheses to match close parentheses-> Invalid
                                                // For example: ())(
        }
        return openCount == 0; // Fully match open parentheses with close parentheses
    }
}


class Solution {
    public boolean checkValidString(String s) {
        int cmin = 0, cmax = 0; // open parentheses count in range [cmin, cmax]
        for (char c : s.toCharArray()) {
            if (c == '(') {
                cmax++;
                cmin++;
            } else if (c == ')') {
                cmax--;
                cmin--;
            } else if (c == '*') {
                cmax++; // if `*` become `(` then openCount++
                cmin--; // if `*` become `)` then openCount--
                // if `*` become `` then nothing happens
                // So openCount will be in new range [cmin-1, cmax+1]
            }
            if (cmax < 0) return false; // Currently, don't have enough open parentheses to match close parentheses-> Invalid
                                        // For example: ())(
            cmin = Math.max(cmin, 0);   // It's invalid if open parentheses count < 0 that's why cmin can't be negative
        }
        return cmin == 0; // Return true if can found `openCount == 0` in range [cmin, cmax]
    }
}





// Recursive Solution
class Solution {
    bool ex(int ind, int openingBracket, string &s){
        if(ind==s.size()) return (openingBracket==0);

        bool ans=false;
        if(s[ind]=='*'){
            ans|=ex(ind+1,openingBracket+1,s); // Add '('
            if(openingBracket) ans|=ex(ind+1,openingBracket-1,s); // Add ')'
            ans|=ex(ind+1,openingBracket,s); //Add Nothing
        }else{
            if(s[ind]=='('){
                ans=ex(ind+1,openingBracket+1,s);
            }else{
                if(openingBracket) ans=ex(ind+1,openingBracket-1,s);
            }
        }

        return ans;
    }

public:
    bool checkValidString(string s) {
        return ex(0,0,s);
    }
};


// Memoization
class Solution {
    bool ex(int ind, int openingBracket, string &s, vector<vector<int>> &dp){
        if(ind==s.size()) return (openingBracket==0);

        if(dp[ind][openingBracket]!=-1) return dp[ind][openingBracket];

        bool ans=false;
        if(s[ind]=='*'){
            ans|=ex(ind+1,openingBracket+1,s,dp);
            if(openingBracket) ans|=ex(ind+1,openingBracket-1,s,dp);
            ans|=ex(ind+1,openingBracket,s,dp);
        }else{
            if(s[ind]=='('){
                ans=ex(ind+1,openingBracket+1,s,dp);
            }else{
                if(openingBracket) ans=ex(ind+1,openingBracket-1,s,dp);
            }
        }

        return dp[ind][openingBracket]=ans;
    }

public:
    bool checkValidString(string s) {
        vector<vector<int>> dp(s.size(), vector<int>(s.size(),-1));
        return ex(0,0,s,dp);
    }
};




// Tabulation
class Solution {
public:
    bool checkValidString(string s) {
        vector<vector<int>> dp(s.size()+1, vector<int>(s.size()+1,0));
        dp[s.size()][0]=1;

        for(int ind=s.size()-1; ind>=0; ind--){
            for(int openingBracket=0; openingBracket<s.size(); openingBracket++){
                bool ans=false;
                if(s[ind]=='*'){
                    ans|=dp[ind+1][openingBracket+1];
                    if(openingBracket) ans|=dp[ind+1][openingBracket-1];
                    ans|=dp[ind+1][openingBracket];
                }else{
                    if(s[ind]=='('){
                        ans|=dp[ind+1][openingBracket+1];
                    }else{
                        if(openingBracket) ans|=dp[ind+1][openingBracket-1];
                    }
                }

                dp[ind][openingBracket]=ans;
            }
        }

        return dp[0][0];
    }
};

------------------------------------------------------------------------------
241. Different Ways to Add Parentheses
Given a string expression of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. You may return the answer in any order.
The test cases are generated such that the output values fit in a 32-bit integer and the number of different results does not exceed 104.

class Solution {
    bool isOperator(char ch) {
        return (ch == '+' || ch == '-' || ch == '*');
    }

    vector<int> getDiffWays(int i, int j, vector<vector<vector<int>>>& dp, string& expression) {

        // Return cached result if already calculated
        if(!dp[i][j].empty()) {
            return dp[i][j];
        }
        
        // If length of the substring is 1 or 2
        // we encounter our base case i.e. a number found.
        int len = j - i + 1;
        if(len <= 2) {
            return dp[i][j] = { stoi(expression.substr(i, len)) };
        }

        // If it is not a number then it is an expression
        // now we try to evaluate every opertor present in it
        vector<int> res;
        for(int ind = i; ind <= j; ind++) {
            if(isOperator(expression[ind])) {
                char op = expression[ind];

                // if char at ind is operator 
                // get all results for its left and right substring using recursion
                vector<int> left = getDiffWays(i, ind - 1, dp, expression);
                vector<int> right = getDiffWays(ind + 1, j, dp, expression);

                // try all options for left & right operand
                // and push all results to the answer
                for(int l : left) {
                    for(int r : right) {
                        if(op == '+') {
                            res.push_back(l + r);
                        }
                        else if(op == '-') {
                            res.push_back(l - r);
                        }
                        else if(op == '*') {
                            res.push_back(l * r);
                        }
                    }
                }
            }
        }
        return dp[i][j] = res;
    }

public:
    vector<int> diffWaysToCompute(string expression) {
        int n = expression.size();
        vector<vector<vector<int>>> dp(n, vector<vector<int>>(n));
        return getDiffWays(0, n - 1, dp, expression);
    }
};

---------------------------------------------------------------------------------------------------------------------------------------------
1249. Minimum Remove to Make Valid Parentheses

###### BOTH AAGE AUR PICHE SE ITERATE ISLIE KARE AS JYADA OPEN BRACE BOTH START AUR END ME HO SKTA THA ((((_ ))     ((()))))))) ----> YA PHIR STACK USE KARO WITH INDEX
class Solution {
public:
    string minRemoveToMakeValid(string s) {
        int n=s.length();
        // Step 1 : Iterate from start
        int count=0; 
        for(int i=0;i<n;++i){
            if(s[i]=='('){ // for open bracket
                ++count;
            }
            else if(s[i]==')'){ // for close bracket
                if(count==0){  // if no. of close brackets > no. of open brackets
                    s[i]='#';
                }
                else{
                    // if matching parentheses found decrease count
                    --count;
                }
            }
        }
        
        // Step 2 : Iterate from end
        count=0;
        for(int i=n-1;i>=0;--i){
            if(s[i]==')'){ // for close bracket
                ++count;
            }
            else if(s[i]=='('){ // for open bracket
                if(count==0){ // if no. of open brackets > no. of close brackets
                    s[i]='#';
                }
                else{
                    // if matching parentheses found decrease count
                    --count;
                }
            }
        }
        
        // Step 3 : Create "ans" by ignoring the special characters '#'
        string ans="";
        for(int i=0;i<n;++i){
            if(s[i]!='#'){ 
                ans.push_back(s[i]);
            }
        }
        return ans;
    }
};



class Solution {
public:
    string minRemoveToMakeValid(string s) {
        
        stack<pair<char,int>>st;
        
        string ans="";
        
        set<int>Intset;
        
        for(int i=0;i<s.size();i++){
            
              if(s[i]==')' || s[i]=='('){
                  
                  if(st.empty()){
                      st.push({s[i],i});
                  }else if(st.top().first=='(' && s[i]==')'){ 
                      st.pop();
                  }else{
                      st.push({s[i],i});
                  }
              }
        }
      
        
        while(!st.empty()){
            int val=st.top().second;
            Intset.insert(val);
            st.pop();
        }
        
        for(int i=0;i<s.size();i++){
            
             if(Intset.find(i)==Intset.end())ans+=s[i];
        }
        
        return ans;
    }
};

--------------------------------------------------------------------------------------------------------------------------------------------------------
1262. Greatest Sum Divisible by Three



class Solution {
public:
     int solve(int i, int curr_sum_rem,vector<int>& nums,vector<vector<int>>&dp){
    if(i>=nums.size()){
        if(curr_sum_rem==0){
            return 0;
        }
        return INT_MIN;
    }
    if(dp[i][curr_sum_rem]!=-1)
        return dp[i][curr_sum_rem];
    int pick =nums[i]+ solve(i+1,(curr_sum_rem+nums[i])%3,nums,dp);    
    int notpick =0+ solve(i+1,curr_sum_rem,nums,dp);  
    return dp[i][curr_sum_rem]=max(pick,notpick);
    
}
    
    int maxSumDivThree(vector<int>& nums) {
        int n =nums.size();
        vector<vector<int>> dp(n,vector<int>(3,-1)); 
        return solve(0,0,nums,dp);
    }
};

class Solution {
public:
    int helper(vector<int>& nums){
        int tsum=0, ones=INT_MAX-40001, twos=INT_MAX-40001;
        
        for(int i=0; i<nums.size(); i++){
            tsum+=nums[i];
            if(nums[i]%3==1){
                twos=min(twos, ones + nums[i]);
                ones=min(ones, nums[i]);
            }
            else if(nums[i]%3==2){
                ones=min(ones, twos + nums[i]);
                twos=min(twos, nums[i]);
            }
        }
        
        if(tsum%3==0) 
            return tsum;
        else if(tsum%3==1) 
            return tsum - ones;
        else    
            return tsum - twos;
    }
    int maxSumDivThree(vector<int>& nums) {
        int n=nums.size();
        if(n==0) return 0;
        return helper(nums);
    }
};

------------------------------------------------------------------------------------------------------------------------------------------
1888. Minimum Number of Flips to Make the Binary String Alternating

You are given a binary string s. You are allowed to perform two types of operations on the string in any sequence:

    Type-1: Remove the character at the start of the string s and append it to the end of the string.
    Type-2: Pick any character in s and flip its value, i.e., if its value is '0' it becomes '1' and vice-versa.

Return the minimum number of type-2 operations you need to perform such that s becomes alternating.

The string is called alternating if no two adjacent characters are equal.

    For example, the strings "010" and "1010" are alternating, while the string "0100" is not.

1888. Minimum Number of Flips to Make the Binary String Alternating
class Solution {
public:
    int minFlips(string s) {
        int n = s.size();
        s += s;
        string s1, s2;
        
        for(int i = 0; i < s.size(); i++) {
            s1 += i % 2 ? '0' : '1';
            s2 += i % 2 ? '1' : '0';
        }
        int ans1 = 0, ans2 = 0, ans = INT_MAX;
        for(int i = 0; i < s.size(); i++) {
            if(s1[i] != s[i]) ++ans1;
            if(s2[i] != s[i]) ++ans2;
            if(i >= n) { //the most left element is outside of sliding window, we need to subtract the ans if we did `flip` before.
                if(s1[i - n] != s[i - n]) --ans1;
                if(s2[i - n] != s[i - n]) --ans2;
            }
            if(i >= n - 1)
                ans = min({ans1, ans2, ans});
        }
        return ans;
    }
};


class Solution {
public:
    int minFlips(string s) {
        int n = s.size();
        int ans1 = 0, ans2 = 0, ans = INT_MAX;
        for(int i = 0; i < 2 * n; i++) {
            if(i < n) s[i] -= '0'; //make '1' and '0' to be integer 1 and 0.
            if(i % 2 != s[i % n]) ++ans1;
            if((i + 1) % 2 != s[i % n]) ++ans2;
            if(i >= n) {
                if((i - n) % 2 != s[i - n]) --ans1;
                if((i - n + 1) % 2 != s[i - n]) --ans2;
            }
            if(i >= n - 1)
                ans = min({ans1, ans2, ans});
        }
        return ans;
    }
};




























































