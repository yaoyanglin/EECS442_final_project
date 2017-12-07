import os

def main():
    count=1
    precesions=[]
    recalls=[]
    for i in range(1,55):
        try:
            f=open('Sample'+str(i)+'/detection.txt')
            file=f.readlines()
            print(file)
            tp=file[2]
            tp=float(tp)
            nbboxes=file[1].split()[0]
            tpfp=int(nbboxes.split(':')[1])
            first=file[0].split('/')[1]
            tpfn=int(first.split('.')[0])
            precesions.append(tp/tpfp)
            recalls.append(tp/tpfn)
        
    print(precesions)
    print(recalls)
    print(sum(precesions)/len(precesions))
    print(sum(recalls)/len(recalls))
main()