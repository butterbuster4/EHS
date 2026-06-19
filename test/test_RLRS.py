from RLRS.RLRS import RLRS
import cv2

# Initialize once
solver = RLRS(
    model_hori_path="./model/hori_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth",
    model_vrti_path="./model/vrti_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth",
    dqn_path="./model/dqn_jigsaw_final.zip",
    pdn_path="./model/best_efficientnetb3.pth"
)

# Call whenever needed
result_img = solver.solve("./test.jpg")
print(solver.best_score)  # Print the best score achieved

if result_img is not None:
    # Save or Display (Remember to convert BGR for OpenCV)
    cv2.imwrite("final_output.jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print("Puzzle Solved!")