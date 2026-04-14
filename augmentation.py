import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import argparse
import torch
import os
import glob
import tqdm

from visualize_one import visualize_from_file


#간단한 알고리즘
#1. 넙죽이를 뒤틀린 넙죽이로 만든다.
#2. 후보점을 찾는다.
#3. 후보점 위에 넙죽이가 올라갈 수 있는지 확인한다.
#4. 올라가지 못하면 2번으로 돌아간다. 너무 많이 실패하면 아쉬운거지...
#5. Point Cloud에 넙죽이를 추가한다.

DEBUG = False

def nub_sqz(mesh, scene_diag):
    ##################################################
    #   원본 넙죽이에게 scale, color jitter, rotation
    #
    #input :mesh - 원본 넙죽이의 mesh
    #       scene_diag - scene의 대각선 길이
    #
    #output:mesh - 뒤틀린 넙죽이의 mesh
    ##################################################
    center = mesh.centroid
    z_min = mesh.bounds[0][2]
    mesh.apply_translation([-center[0],-center[1],-z_min])
    
    im = np.random.uniform(0,1)
    scale_ratio = 0.025 + 0.175 * (im ** 2) # Scale은 큰 것 보다 작은게 더 자주 발생하도록 함.
    
    sqz_ratio = np.random.uniform(0.5,1.5,size=3)
    angle = np.random.uniform(-180,180,size=3)
    
    curr_nub_size = np.max(mesh.extents)
    new_scale = (scale_ratio * scene_diag / curr_nub_size) * sqz_ratio
    mesh.apply_scale(new_scale)
    
    angles = np.eye(4)
    angles[:3,:3] = R.from_euler('xyz', angle, degrees=True).as_matrix() #덤벼라 짐벌락
    mesh.apply_transform(angles)
    return mesh

def mesh2pc(mesh,n):
    ##################################################
    #   넙죽이 mesh를 pc로 변환
    #   color jitter는 여기서 하도록.
    #
    #input :mesh - 뒤틀린 넙죽이 mesh
    #       n - 넙죽이의 pc 개수
    #
    #output:pc - 넙죽이의 pc
    ##################################################
    point, face = trimesh.sample.sample_surface(mesh,n)
    normal = mesh.face_normals[face]
    cv = mesh.visual.to_color()
    cv.mesh = mesh
    color = cv.face_colors[face][:,:3]
    jitter = np.random.normal(-20,20,size=3)
    color_jitter = np.clip(color+jitter,0,255)
    return point.astype(np.float32), color_jitter.astype(np.uint8), normal.astype(np.float32)


def candy(scene_point, scene_normal, threshold = 0.8):
    ##################################################
    #   넙죽이가 들어갈 수 있을 후보군을 법선 벡터로만 확인
    # 
    #input :scene_point - scene의 point(xyz) 어라? 필요 없을지도
    #       scene_normal - scene의 법선벡터
    #
    #output:point (N,) - 후보 점들의 index
    ##################################################
    mask = scene_normal[:,2] > threshold
    candy = scene_point[mask]
    
    x_min, x_max = np.percentile(candy[:,0],[5,95])
    y_min, y_max = np.percentile(candy[:,1],[5,95])
    z_min, z_max = np.percentile(candy[:,2],[5,95])
    
    bound_mask = (candy[:,0] > x_min) & (candy[:,0] < x_max) & \
                 (candy[:,1] > y_min) & (candy[:,1] < y_max) & \
                 (candy[:,2] > z_min) & (candy[:,2] < z_max)
    candy = candy[bound_mask]
    
    if len(candy) == 0:
        candy = scene_point[mask]
    return candy

def check(nub_point, scene_tree,scene):
    ##################################################
    #check_coll과 check_sup 부르기
    ##################################################
    return check_coll(nub_point, scene_tree) and check_sup(nub_point,scene_tree) and check_bound(nub_point,scene) #좀 허센데

def check_bound(nub_point, scene):
    ##################################################
    #   넙죽이가 배경 밖으로 탈출하는 것 막기
    ##################################################
    scene_min = np.min(scene, axis=0)
    scene_max = np.max(scene, axis=0)
    nub_min = np.min(nub_point, axis=0)
    nub_max = np.max(nub_point, axis=0)
    
    return np.all(nub_min >= scene_min) and np.all(nub_max <= scene_max)


def check_coll(nub_point, scene_tree):
    ##################################################
    #   넙죽이와 기존 PC간의 충돌 확인인 
    #
    #   넙죽이의 bounding box 내부에 pc존재여부로 판단함
    #
    #input :nub_point - 넙죽이의 pc
    #       scene_tree - scene의 pc tree
    #
    #output:res - 충돌 여부
    ##################################################
    min_nub = np.min(nub_point,axis=0)
    max_nub = np.max(nub_point,axis=0)
    
    min_nub += 0.02
    max_nub -= 0.02
    
    mask = (scene_tree.data[:,0] > min_nub[0]) & (scene_tree.data[:,0] < max_nub[0]) & \
           (scene_tree.data[:,1] > min_nub[1]) & (scene_tree.data[:,1] < max_nub[1]) & \
           (scene_tree.data[:,2] > min_nub[2]) & (scene_tree.data[:,2] < max_nub[2])
    
    return np.sum(mask) <= 10



def check_sup(nub_point, scene_tree):
    ##################################################
    #   넙죽이와 기존 PC간의 지지대가 충분한지 확인
    #   단 나중에 좀 더 넙죽이를 벽에 붙일 수 있게 수정 요망.
    #
    #input :nub_point - 넙죽이의 pc
    #       scene_tree - scene의 pc tree
    #
    #output:res - 지지대 여부
    ##################################################
    z_min = np.min(nub_point[:,2])
    mask = nub_point[:,2] <= z_min + 0.05
    bottom_nub_point = nub_point[mask]
    
    d, _ = scene_tree.query(bottom_nub_point, distance_upper_bound=0.05)
    sup_count = np.sum(d != np.inf)
    return sup_count > 0.01 * len(bottom_nub_point)


def merge_nub(scene_data, nub_xyz, nub_color, nub_normal, nub_id):
    ##################################################
    #   PC랑 넙죽이 PC 합치기기
    #
    #input :scene_data - 기존 scene의 data
    #       nub-xxx - 넙죽이의 데이터
    #
    #output:pc - 합친 point cloud
    ##################################################
    scene_data['xyz'] = np.vstack((scene_data['xyz'],nub_xyz))
    scene_data['rgb'] = np.vstack((scene_data['rgb'],nub_color))
    scene_data['normal'] = np.vstack((scene_data['normal'],nub_normal))
    label = np.full(len(nub_xyz),nub_id,dtype=np.int32)
    scene_data['instance_labels'] = np.hstack((scene_data['instance_labels'],label))
    return scene_data

def find_place(nub_xyz,scene_xyz,scene_normal,scene_tree,nub_center,scene_label):
    ##################################################
    #   넙죽이를 뒤틀고, scene에 삽입
    #
    #input :nub_data - 기존 넙죽이의 data
    #       scene-xxx - scene의 데이터
    #
    #output:pc - 합친 point cloud
    ##################################################
    
    
    candidate_point = candy(scene_xyz,scene_normal) #넙죽이 위의 넙죽이 허용
    
    nub_anchor = [np.mean(nub_xyz[:,0]),np.mean(nub_xyz[:,1]),np.min(nub_xyz[:,2])] # 이거 z만 다름.
    z_min = np.min(candidate_point[:,2])
    z_max = np.max(candidate_point[:,2])
    
    
    for _ in range(1000):
        #c_idx = np.random.randint(0,len(candidate_point))
        #바닥에 너무 많이 깔리는 문제가 있어서 전체 랜덤이 아니라, 일단 z에 대해서 랜덤으로 뽑는 방식으로 바꿈.
        #rand_z = np.random.uniform(z_min,z_max)
        im = np.random.uniform(0,1)
        rand_z = z_max - (z_max - z_min) * (im ** 4)
        z_space = np.abs(candidate_point[:,2] - rand_z)
        mask = z_space < 0.1
        mask_candy = candidate_point[mask]
        if(len(mask_candy) == 0):
            continue
        c_idx = np.random.randint(0,len(mask_candy))
        candy_point = mask_candy[c_idx]
        
        if(len(nub_center) > 0):
            d = np.linalg.norm(nub_center - candy_point,axis=1)
            if np.any(d<1):
                continue
        
        if DEBUG == True:
            print(rand_z)
        t = candy_point - np.array(nub_anchor)
        translate_nub = nub_xyz + t
        if check(translate_nub, scene_tree, scene_xyz):
            return translate_nub, candy_point
    return None, None

def generate_nubscene(scene_path, nub_path, output_path,output_ply_path):
    ##################################################
    #   최대 5명의 넙죽이를 scene에 합성
    #
    #input :nub_path - 넙죽이의 주소 - 고정
    #       scene_path - scene의 주소
    #
    #output:성공 여부
    ##################################################
    data = torch.load(scene_path,weights_only=False)
    
    data['instance_labels'] = np.zeros(len(data['xyz']),dtype=np.int32)
    
    scene_min = np.min(data['xyz'],axis=0)
    scene_max = np.max(data['xyz'],axis=0)
    scene_diag = np.linalg.norm(scene_max-scene_min)
    if DEBUG == True:
        print("scene_min", scene_min)
        print("scene_max", scene_max)
        print("scene_diag", scene_diag)
    nub_n = np.random.randint(1,6)
    if DEBUG:
        nub_n = 5
    nub_center = []
    i = 1
    while True:    
        scene_tree = cKDTree(data['xyz'])
        nub_mesh = trimesh.load(nub_path, force='mesh')
        
        nub_mesh = nub_sqz(nub_mesh, scene_diag)
        p, c, n = mesh2pc(nub_mesh,10000)
        
        tp, cp = find_place(p,data['xyz'],data['normal'],scene_tree,nub_center,data['instance_labels'])
        if tp is not None:
            data = merge_nub(data,tp,c,n,i)
            nub_center.append(cp)
            if DEBUG == True:
                print(i,"th nub is create")
            i += 1
            if i == nub_n+1:
                break
    
    if output_ply_path is not None:
        pc = trimesh.PointCloud(data['xyz'], colors=data['rgb'])
        pc.export(output_ply_path)
        
    output_data = {
        'xyz': data['xyz'],
        'rgb': data['rgb'],
        'normal': data['normal'],
        'instance_labels': data['instance_labels']
    }
    
    np.save(output_path,output_data)
    

    
    
def process_all(data_dir, output_dir):
    ########################################################
    #   모든 pth에 대해서 돌리는 코드
    #   제미나이한테 맡겨서 문제 있을 수도 이슴
    ########################################################
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'npy'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'png'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ply'), exist_ok=True)
    pth_files = glob.glob(os.path.join(data_dir, '*.pth'))
    
    print(f"총 {len(pth_files)}개의 .pth 파일을 찾았습니다. 처리를 시작합니다.")
    
    success_count = 0
    fail_count = 0
    
    for idx, pth_path in enumerate(tqdm.tqdm(pth_files)):
        file_name = os.path.basename(pth_path)
        base_name = os.path.splitext(file_name)[0]
        # 원본과 구분하기 위해 _aug 접미사 추가
        output_npy_path = os.path.join(output_dir, 'npy', f"{base_name}_aug.npy")
        output_png_path = os.path.join(output_dir, 'png', f"{base_name}_aug.png")
        output_ply_path = os.path.join(output_dir, 'ply', f"{base_name}_aug.ply")
        glb_path = 'assets/sample.glb'
        
        print(f"[{idx+1}/{len(pth_files)}] 처리 중: {file_name} -> {base_name}_aug.npy", end=" ... ")
        
        try:
            generate_nubscene(pth_path, glb_path, output_npy_path,output_ply_path)
            if DEBUG == True:
                visualize_from_file(
                    data_npy_path=output_npy_path,
                    output_path=output_png_path,
                    max_points=300000,
                    point_size=3.0,
                    views=("front",)
                )
                
            print("성공")
            success_count += 1
        except Exception as e:
            print(f"실패 (에러: {e})")
            fail_count += 1
            
    print(f"\n처리 완료! (성공: {success_count}, 실패: {fail_count})")

    

def main():
    parser = argparse.ArgumentParser(description="Visualize original scene point cloud.")
    parser.add_argument("--data-pth", type=str,default=None, required=False, help="Path to one *_aug.npy scene file")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    nub_pth = 'assets/sample.glb'
    DEBUG = args.debug
    if not DEBUG:
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
    
    if args.data_pth is None:
        data_dir = 'data'
        output_dir = 'output'
        process_all(data_dir, output_dir)
        return 
    
    test_output = 'test_output/out'
    nub_pth = 'assets/sample.glb'
    generate_nubscene(args.data_pth,nub_pth,test_output,'test_output/out.ply')
    visualize_from_file(
        data_npy_path='test_output/out.npy',
        output_path='test_output/out.png',
        max_points=300000,
        point_size=3.0,
        views=("front", "back", "left", "right", "top", "bottom")
    )
    print(f"Saved new scene to: {test_output}")


if __name__ == "__main__":
    main()
    


