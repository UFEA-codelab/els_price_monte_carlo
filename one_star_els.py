import time
import numpy as np


def simulate_stock(n,d,rfr,vol,div,dt=1/250): # 주식 수익률 시뮬레이션 행렬 만들기
    diffusion, drift = vol*np.sqrt(dt), (rfr - div - 0.5 * vol**2)*dt # diffusion, drift rate 반영
    ret=np.random.normal(size=(n,d-1)) # 랜덤 정규분포 행렬 생성
    return_array=np.full((n,1),1) # 종목별 수익률 array 생성
    for i in range(d-1): # for loop 돌면서 drift, diffusion 적용한 수익률 붙이기
        return_array=np.append(return_array,(return_array[:,-1]*(np.exp(drift + diffusion * ret[:,i]))).reshape(-1,1),axis=1)
    return return_array # 1000번 750 기간의 주식 시뮬레이션 반환


def check_redemp_time(return_array,barrier,num_redemp): # 상환 시간 체크 함수
    time_redemp=np.full((return_array.shape[0],1),num_redemp+1) # 상환 시간 행렬 생성
    for i in range(num_redemp): # 상환횟수만큼 for문
        tmp=return_array[:,schedule[i]-1]-barrier[i] # 각 상환시점 별로 수익률 - barrier 값 저장
        tmp[tmp>=0]=i # 뺀 값이 0 이상이면 현재 시점 저장
        tmp[tmp<0]=num_redemp+1 # 뺀 값이 0 미만이면 만기 시점 저장
        time_redemp=np.append(time_redemp,tmp.reshape(-1,1),axis=1)
    time_redemp=np.min(time_redemp, axis=1) # 만기 횟수만큼 돌면서 가장 작은 값 반환 (한 번이라도 0 미만 이었으면 가장 앞 시점 값 저장됨)
    return time_redemp # 1000개 종목의 각 상환 시점 array


def check_knock_in_barrier(return_array,ki_barrier): # 해당 시뮬레이션이 낙인 배리어 쳤는지 체크하는 함수
    min_stock=np.min(return_array,axis=1) # 주가 최저점을 저장하는 array
    return np.where(min_stock<ki_barrier,1,0) # 최저점이 낙인 배리어 아래이면 1, 아니면 0 반환


def get_one_els_value(return_last,time_redemp,ki_array,coupon,schedule,rfr):
    sum_price,prob_array=0,np.zeros(len(schedule)+2) # 시뮬레이션 별로 els 가격의 총 합, 각 지점 별 확률
    for i in range(len(time_redemp)): # 시뮬레이션 별로 지점 별 확률 loop
        t = int(time_redemp[i]) # 시뮬레이션 별 배리어 밑으로 수익률이 하락했던 최초 지점
        if t<len(coupon): # 만기 이전에 하락했으면 아래의 로직으로 더함
            discount_factor = np.exp(-rfr * schedule[t] / 250)
            sum_price+=(1+coupon[t])*discount_factor
            prob_array[t]+=1
        else: # 만기까지 갔다면 두 가지로 나뉨
            t=min(t,len(coupon)-1)
            discount_factor = np.exp(-rfr * schedule[t] / 250)
            if ki_array[i]==1: # 낙인 배리어 밑으로 하락했으면 아래 로직
                sum_price+=return_last[t]*discount_factor
                prob_array[-1]+=1
            else: # 낙인 배리어 밑으로 하락한 적 없으면 아래 로직
                sum_price+=coupon[-1]*discount_factor
                prob_array[-2]+=1
    els_value,redemp_porb=sum_price / len(time_redemp), prob_array / len(time_redemp)
    return els_value,redemp_porb


start=time.time()
num_redemp,ki_barrier,rfr=6,0.6,0.03 # 상환횟수, 낙인 배리어, 무위험이자율
barrier=np.array([0.9, 0.9, 0.85, 0.85, 0.8, 0.8]) # 조기/만기상환 배리어
schedule = np.arange(1, num_redemp + 1) * 125 #
coupon = np.arange(1, num_redemp + 1) * 0.05

n_iteration=10000
return_array1=simulate_stock(n_iteration,num_redemp*125,rfr,0.3,0) # 인풋: 시뮬레이션 횟수, 기간, 무위험이자율, 변동성, 배당율
time_redemp1=check_redemp_time(return_array1,barrier,num_redemp) # 인풋: 수익률 행렬, 배리어 벡터, 상환 횟수 값
ki_array1=check_knock_in_barrier(return_array1,ki_barrier) # 인풋: 수익률 행렬, 낙인 배리어 값
els_price,prob=get_one_els_value(return_array1[:,-1],time_redemp1,ki_array1,coupon,schedule,rfr)
print(els_price,prob)
print("opt version: %s iteration, %s seconds" % (n_iteration,time.time()-start))
