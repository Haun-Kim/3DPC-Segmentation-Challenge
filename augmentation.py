import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import argparse
import torch
import os
import glob
import tqdm
import matplotlib.colors as mcolor

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
    mesh = mesh.copy()
    center = mesh.centroid
    z_min = mesh.bounds[0][2]
    mesh.apply_translation([-center[0],-center[1],-z_min])
    
    im = np.random.uniform(0,1)
    scale_ratio = 0.025 + 0.175 * (im ** 1.0)
    
    sqz_ratio = np.random.uniform(0.5,1.5,size=3)
    angle = np.random.uniform(-180,180,size=3)
    
    curr_nub_size = np.max(mesh.extents)
    final_ratio = np.clip(scale_ratio * sqz_ratio,0.025,0.2)
    new_scale = final_ratio * (scene_diag / curr_nub_size)
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
    
    color_jitter = color_jittering(color)
    
    return point.astype(np.float32), color_jitter.astype(np.uint8), normal.astype(np.float32)

def color_jittering(color):
    ##################################################
    #   color jitter
    #   input : color
    #   output: jittered color
    ##################################################
    rgb_f = color.astype(np.float32) / 255.0

    h_shift = np.random.uniform(-0.15, 0.15)
    s_scale = np.random.uniform(0.65, 1.35)
    v_scale = np.random.uniform(0.75, 1.45)

    if np.random.rand() < 0.20:
        h_shift = np.random.uniform(-0.5, 0.5)

    hsv = mcolor.rgb_to_hsv(rgb_f)
    hsv[:, 0] = (hsv[:, 0] + h_shift) % 1.0
    hsv[:, 1] = np.clip(hsv[:, 1] * s_scale, 0.0, 1.0)
    hsv[:, 2] = np.clip(hsv[:, 2] * v_scale, 0.0, 1.0)

    rgb_out = mcolor.hsv_to_rgb(hsv)
    return (rgb_out * 255.0).astype(np.uint8)
    
   
    


def candy(scene_point, scene_normal,scene_label, threshold = 0.65):
    ##################################################
    #   넙죽이가 들어갈 수 있을 후보군을 법선 벡터로만 확인
    # 
    #input :scene_point - scene의 point(xyz) 어라? 필요 없을지도
    #       scene_normal - scene의 법선벡터
    #
    #output:point (N,) - 후보 점들의 index
    ##################################################
    
    mask_bg = scene_label == 0
    mask_nub = scene_label > 0    
    mask = scene_normal[:,2] > threshold
    
    candy_bg = scene_point[mask_bg & mask]
    candy_nub = scene_point[mask_nub& mask]
    
    if len(candy_nub) > 0:
        n = max(1, len(candy_nub) // 200)
        mask_nub_n = torch.randperm(len(candy_nub),device=scene_point.device)[:n]
        candy_nub = candy_nub[mask_nub_n]

    
    candy = torch.cat([candy_bg,candy_nub],dim=0)
    #일반적인 상황에서 len(candy) != 0
    q_low = torch.quantile(candy,0.05,dim=0)
    q_high = torch.quantile(candy,0.95,dim=0)
    
    bound_mask = (candy[:,0] >= q_low[0]) & (candy[:,0] <= q_high[0]) & \
                 (candy[:,1] >= q_low[1]) & (candy[:,1] <= q_high[1]) & \
                 (candy[:,2] >= q_low[2]) & (candy[:,2] <= q_high[2])
    candy = candy[bound_mask]
    
    if len(candy) == 0:
        candy = scene_point[mask]
    return candy

def check_bound(nub_point, scene_min,scene_max):
    ##################################################
    #   넙죽이가 배경 밖으로 탈출하는 것 막기
    ##################################################
    nub_min = nub_point.min(dim=0)[0]
    nub_max = nub_point.max(dim=0)[0]
    
    return torch.all(nub_min >= scene_min) and torch.all(nub_max <= scene_max)


def get_AABB(nub_xyz, scene_xyz, scene_normal, threshold):
    min_nub = nub_xyz.min(dim=0)[0] - threshold
    max_nub = nub_xyz.max(dim=0)[0] + threshold
    

    mask = (scene_xyz[:,0] > min_nub[0]) & (scene_xyz[:,0] < max_nub[0]) & \
           (scene_xyz[:,1] > min_nub[1]) & (scene_xyz[:,1] < max_nub[1]) & \
           (scene_xyz[:,2] > min_nub[2]) & (scene_xyz[:,2] < max_nub[2])
    return scene_xyz[mask],scene_normal[mask]


def check_coll(nub_xyz,nub_normal,AABB_scene,AABB_normal):
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
    if len(AABB_scene) == 0:
        return False
    d = torch.cdist(AABB_scene,nub_xyz)
    dist,idx = d.min(dim=1)
    
    
    val_AABB = AABB_scene
    val_nub_xyz = nub_xyz[idx]
    val_nub_normal = nub_normal[idx]
    
    V_scene = val_AABB - val_nub_xyz
    V_unit = V_scene / (torch.linalg.norm(V_scene,dim=1,keepdim=True) + 1e-8)
    
    dot = torch.sum(V_unit * val_nub_normal,dim=1)

    mask_inside = (dot < -0.05) & (dist < 0.1)
    if mask_inside.sum() > 10 :
        return True
    
    dist2, idx2 = d.transpose(0,1).min(dim=1)
    val_AABB = AABB_scene[idx2]
    val_scene_normal = AABB_normal[idx2]
    
    V_nub = nub_xyz - val_AABB
    V_nub_unit = V_nub / (torch.linalg.norm(V_nub,dim=1,keepdim=True) + 1e-8)
    
    dot = torch.sum(V_nub_unit * val_scene_normal,dim=1)   
    mask_inside_scene = (dot < -0.05) & (dist2 < 0.1)
    if mask_inside_scene.sum() > 100 :
        return True
    
    return False

    

def check_sup(nub_point, AABB_scene):
    ##################################################
    #   넙죽이와 기존 PC간의 지지대가 충분한지 확인
    #   단 나중에 좀 더 넙죽이를 벽에 붙일 수 있게 수정 요망.
    #
    #input :nub_point - 넙죽이의 pc
    #       scene_tree - scene의 pc tree
    #
    #output:res - 지지대 여부
    ##################################################
    z_min = nub_point[:,2].min()
    
    mask = nub_point[:,2] <= z_min + 0.05
    bottom_nub_point = nub_point[mask]
    if len(AABB_scene) == 0:
        return False
    d = torch.cdist(bottom_nub_point, AABB_scene)
    dist = d.min(dim=1)[0]
    sup_ratio= (dist < 0.03).float().sum() / len(bottom_nub_point)
    return sup_ratio > 0.25


def rand_downsampling(candy_point,scene_diag):
    if len(candy_point) == 0:
        return candy_point
    voxel_size = float(np.clip(scene_diag * 0.04, 0.1, 0.25))
    idx = torch.randperm(candy_point.size(0),device=candy_point.device)
    rand_point = candy_point[idx]
    
    voxel_coord = torch.floor(rand_point / voxel_size).int()
    v, inv = torch.unique(voxel_coord, dim=0, return_inverse=True)
    
    idx = torch.arange(inv.size(0),device=inv.device)
    ret_idx = torch.zeros(v.size(0),dtype=torch.long,device=inv.device)
    ret_idx.scatter_(0,inv,idx)
    return rand_point[ret_idx]

def weigh_sampling(candidate_point, bin_size = 0.05,alpha = 1.5):
    z_coord = candidate_point[:,2]

    z_bin = torch.floor(z_coord / bin_size).long()
    z_bin = z_bin - z_bin.min()

    bin_count = torch.bincount(z_bin)
    w = 1.0 / (bin_count[z_bin].float() ** alpha)
    n = min(1000, len(w))
    idx = torch.multinomial(w, n, replacement=False)
    return candidate_point[idx]

def z_sampling(candidate_point,z_min,z_max,b, bin_size = 0.05):
    im = torch.rand(b,device=candidate_point.device)
    rand_z = z_max - (z_max - z_min) * (im)
    
    z_space = torch.abs(candidate_point[:,2].unsqueeze(0) - rand_z.unsqueeze(1))
    dist,idx = z_space.min(dim=1)
    mask = dist < 0.1
    b_candidate_point = candidate_point[idx[mask]]
    return b_candidate_point


def find_place(nub_xyz,nub_normal,scene_xyz,scene_normal,scene_label,scene_diag,nub_center):
    ##################################################
    #   넙죽이를 뒤틀고, scene에 삽입
    #
    #input :nub_data - 기존 넙죽이의 data
    #       scene-xxx - scene의 데이터
    #
    #output:pc - 합친 point cloud
    ##################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nub_anchor = torch.tensor([nub_xyz[:,0].mean(),nub_xyz[:,1].mean(),nub_xyz[:,2].min()],dtype=torch.float32).to(device)
    
    candidate_point = candy(scene_xyz,scene_normal,scene_label)
    if len(candidate_point) == 0:
        return None, None
    scene_min = scene_xyz[:,2].min()
    floor_threhold = scene_min + 0.15
    
    floor_nub = False
    for c in nub_center:
        if c[2] < floor_threhold:
            floor_nub = True
            break
    
    if floor_nub:
        m = candidate_point[:,2] > floor_threhold
        candidate_point = candidate_point[m]
        if len(candidate_point) == 0:
            return None, None
    v_canidate_point = rand_downsampling(candidate_point,scene_diag)
    b_candidate_point = weigh_sampling(v_canidate_point)
    valid_proposals =[]
    K_MAX = 20
    
    
    z_min = candidate_point[:,2].min()
    z_max = candidate_point[:,2].max()
    scene_min = scene_xyz.min(dim=0)[0]
    scene_max = scene_xyz.max(dim=0)[0]
    
    b = 256
    for _ in range(1024 // b):
        #바닥에 너무 많이 깔리는 문제가 있어서 전체 랜덤이 아니라, 일단 z에 대해서 랜덤으로 뽑는 방식으로 바꿈.
        b_candidate_point = torch.unique(b_candidate_point,dim=0)
        if(len(b_candidate_point) == 0):
            continue
        
            
        t = b_candidate_point - nub_anchor
        t[:,2] += 0.05
        rand_idx = torch.randperm(len(t))
        for i in rand_idx:
            translate_nub = nub_xyz + t[i]
            AABB_scene, AABB_normal = get_AABB(translate_nub, scene_xyz,scene_normal, 0.2)
            
            if check_bound(translate_nub,scene_min,scene_max) == False:
                continue            
            if check_coll(translate_nub, nub_normal, AABB_scene,AABB_normal) :
                continue
            
            flag = False
            for _ in range(15):
                if check_sup(translate_nub, AABB_scene):
                    if check_coll(translate_nub, nub_normal, AABB_scene,AABB_normal) == False:
                        flag = True
                    break
                
                translate_nub[:,2] -=  0.01 
                
                if check_coll(translate_nub, nub_normal, AABB_scene,AABB_normal) == True:
                    break
                
            if flag:
                valid_proposals.append((translate_nub.cpu().numpy(), b_candidate_point[i].cpu().numpy()))
            if len(valid_proposals) > K_MAX:
                break
        if len(valid_proposals) > K_MAX:
                break
    
    if len(valid_proposals) == 0:
        return None, None
    if len(nub_center) > 0:
        best_proposal = None
        max_dist = -1
        for trans_nub, cand_pt in valid_proposals:
            # 기존 넙죽이들의 XY 중심과의 거리 계산
            dists = [np.linalg.norm(cand_pt[:2] - np.array(c)[:2]) for c in nub_center]
            min_dist = min(dists)
            if min_dist > max_dist:
                max_dist = min_dist
                best_proposal = (trans_nub, cand_pt)
        return best_proposal[0], best_proposal[1]
    else:
        idx = np.random.choice(len(valid_proposals))
        return valid_proposals[idx][0], valid_proposals[idx][1]

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
    
    scene_min = np.percentile(data['xyz'],5,axis=0)
    scene_max = np.percentile(data['xyz'],95,axis=0)
    scene_diag = np.linalg.norm(scene_max-scene_min)
    if DEBUG == True:
        print("scene_min", scene_min)
        print("scene_max", scene_max)
        print("scene_diag", scene_diag)
    nub_n = np.random.randint(1,6)
    if DEBUG:
        nub_n = 5
    nub_center = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    max_try = 10000
    for i in range(1,nub_n+1):
        scene_xyz = torch.from_numpy(data['xyz']).to(device)
        scene_normal = torch.from_numpy(data['normal']).to(device)
        scene_label = torch.from_numpy(data['instance_labels']).to(device)
        nub_mesh = trimesh.load(nub_path, force='mesh')
        
        for k in range(max_try):    
            nub_sqz_mesh = nub_sqz(nub_mesh, scene_diag)
            num_nub_pc = np.random.randint(12000,20000)
            p, c, n = mesh2pc(nub_sqz_mesh,num_nub_pc)
            nub_xyz = torch.from_numpy(p).to(device)
            nub_normal = torch.from_numpy(n).to(device)
            tp, cp = find_place(nub_xyz,nub_normal,scene_xyz,scene_normal,scene_label,scene_diag,nub_center)
            if tp is not None:
                data = merge_nub(data,tp,c,n,i)
                nub_center.append(cp)
                scene_xyz = torch.from_numpy(data['xyz']).to(device)
                scene_normal = torch.from_numpy(data['normal']).to(device)
                scene_label = torch.from_numpy(data['instance_labels']).to(device)
                if DEBUG == True:
                    print(i,"th nub is create")
                break
        if(k==max_try-1):
            raise RuntimeError(f"{i-1}번째 넙죽이 배치 공간을 찾지 못했습니다.")
    
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
    

    
    
def process_all(data_dir, output_dir, num_copy = 1):
    ########################################################
    #   모든 pth에 대해서 돌리는 코드
    #   제미나이한테 맡겨서 문제 있을 수도 이슴
    ########################################################
    os.makedirs(output_dir, exist_ok=True)
    if DEBUG == True:
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
        if DEBUG == True:
            output_npy_path = os.path.join(output_dir, 'npy', f"{base_name}_aug.npy")
            output_png_path = os.path.join(output_dir, 'png', f"{base_name}_aug.png")
            output_ply_path = os.path.join(output_dir, 'ply', f"{base_name}_aug.ply")
        else:
            output_npy_path = os.path.join(output_dir, f"{base_name}.npy")
            
        glb_path = 'assets/sample.glb'
        
        print(f"[{idx+1}/{len(pth_files)}] 처리 중: {file_name} -> {base_name}_aug.npy", end=" ... ")
        
        try:
            for i in range(1,1+num_copy):
                if DEBUG:
                    output_npy_path_ = os.path.join(output_dir, 'npy', f"{base_name}_{i}.npy")
                    output_png_path_ = os.path.join(output_dir, 'png', f"{base_name}_{i}.png")
                    output_ply_path_ = os.path.join(output_dir, 'ply', f"{base_name}_{i}.ply")
                else:
                    output_npy_path_ = os.path.join(output_dir, f"{base_name}_{i}.npy")
                    output_ply_path_ = None
                    
                generate_nubscene(pth_path, glb_path, output_npy_path_,output_ply_path_)
                
                if DEBUG == True:
                    visualize_from_file(
                        data_npy_path=output_npy_path_,
                        output_path=output_png_path_,
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
    parser.add_argument("--data-pth", type=str,default=None, required=False, help="Path to dir of dply data")
    parser.add_argument("--data-ply", type=str,default=None, required=False, help="Path to one ply")
    parser.add_argument("--out-pth", type=str,default=None, required=False, help="Path to output")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num-scene", type=int,default=1, required=False, help="number of scenes per one ply file")
    
    args = parser.parse_args()
    nub_pth = 'assets/sample.glb'
    global DEBUG
    DEBUG = args.debug
    n = args.num_scene
    
    if args.data_ply is None:
        data_dir = args.data_pth
        output_dir = args.out_pth
        process_all(data_dir, output_dir,n)
        return 
    test_output_dir = 'test_output'
    os.makedirs(test_output_dir, exist_ok=True) # 추가
    test_output = 'test_output/out'
    nub_pth = 'assets/sample.glb'
    generate_nubscene(args.data_ply,nub_pth,test_output,'test_output/out.ply')
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
    


