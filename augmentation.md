우리의 목표! 데이터 증강 알고리즘 만들기!

기본적인 규칙
1. 본 데이터는 MultiScan의 장면과 Nubzuki의 합성을 통해 만들어진다.
2. MultiScan의 모든 Point들은 Backgruond로 평가된다.
3. 넙죽이 데이터는 sample.glb파일로 존재한다.
4. 이제 넙죽이는 최소 1명, 최대 5명이 존재한다.(명?)
5. 넙죽이의 scale ratio는 0.0025~0.2까지 랜덤이다.
6. 넙죽이는 bounding box기준으로 다른 물체 위에도 존재할 수 있으면, 일부 튀어나오는 형태도 허용된다.
7. 겹침은 최소화 해야한다.(단 이건 실제 데이터들을 보면서 살펴봐야할듯)
8. 넙죽이는 몇몇 변환을 거칠 수 있다.
    -  Anisotropic scaling: x-,y-,z-axis로 (0.5,1.5)
    -  Affine transform: x-,y-,z-axis로 (-180,180)
    -  Color map jittering: 이건 뭐냐
9. Format은 .npy파일로 주어진다. field는 다음과 같다.
    - xyz: float32, shape (N, 3)
    - rgb: uint8, shape (N, 3)
    - normal: float32, shape (N, 3)
    - instance_labels: int32, shape (N,) (0 for background, positive IDs for inserted instances)
10. output은 다음과 같다.
    - Predicted_label $$\hat{y} : shape = (N,)$$
11. 현제 코드는 다음과 같은 문제를 갖고 있다.
    - 넙죽이가 생성되지 않을 수도 있기에 


추가로 알아낸 사실
1. test_0000.npy파일에는 추가로 is_mesh라는 field가 주어지는데, 이는 배경일때 False, 나머지일때 True이다. 다만 딱히 mesh에 대한 정보는 없는걸 보면(normal제외) 아마 무언가의 잔제가 아니지 않나 생각된다.

todo.
1. 일단 기본적인 검토. 아직 특정 데이터에 대해서만 확인함
2. 현제 넙죽이 위에 넙죽이를 허용하지 않으나 이거 된다고 함. 수정 할것
3. 현제 저장하는 데이터는 그냥 다 저장하고 있음. 필요한 key 제외하고는 다 del할것