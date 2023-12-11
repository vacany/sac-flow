# class SE3_lietorch(torch.nn.Module):
#     def __init__(self, BS=1, device='cpu'):
#         super().__init__()
#         self.translation = torch.nn.Parameter(torch.zeros((BS, 3), requires_grad=True, device=device))
#         self.quaternions = torch.zeros((BS, 4), device=device)
#         self.quaternions[:, -1] = 1
#         self.quaternions = torch.nn.Parameter(self.quaternions, requires_grad=True)

#         # self.quaternions.requires_grad_(True)

#     def forward(self, pc):

#         lietorch_SE3 = SE3(torch.cat((self.translation, self.quaternions), dim=1))
#         transform_matrix = lietorch_SE3.matrix()

#         pc_to_transform = torch.cat([pc, torch.ones((len(pc), pc.shape[1], 1), device=pc.device)], dim=2)
#         deformed_pc = torch.bmm(pc_to_transform, transform_matrix)[:, :, :3]

#         return deformed_pc
