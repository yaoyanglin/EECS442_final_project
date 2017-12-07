def main():

    precesions=[]
    recalls=[]

    for i in range(1,100):
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
            try:
                precesions.append(tp/tpfp)
                recalls.append(tp/tpfn)
            except:
                pass
        except:
            pass
    print(precesions)
    print(recalls)
    print(sum(precesions)/len(precesions))
    print(sum(recalls)/len(recalls))

main()