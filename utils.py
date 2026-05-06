import cv2

def draw_counts(frame, cars, people):
    # Display car count
    cv2.putText(frame, f"Cars: {cars}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    # Display people count
    cv2.putText(frame, f"People: {people}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    return frame