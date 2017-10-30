library(XML)
library(dplyr)
library(data.table)
setwd("/home/akash/github/face_segmentation/data/ANNOTATION")

Mode <- function(x, na.rm = TRUE) {
  if(na.rm){
    x = x[!is.na(x)]
  }
  
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

get_temp <- function(file_path){
  doc <- xmlParse(file_path)
  temp_list <- xmlToList(doc)
  temp_names <- names(temp_list)[!(names(temp_list) %in% "object")]
  data <- data.frame(temp_list[[temp_names[1]]])
  names(data) <- temp_names[1]
  for (j in c(2:length(temp_names))){
    i = temp_names[j]
    temp_data <- data.frame(temp_list[[i]])
    names(temp_data) <- i
    if (length(temp_list[[i]]) > 1){
      temp_data <- as.data.frame(temp_list[[i]])
      names(temp_data) <- names(temp_list[[i]])
    }
    data <- bind_cols(data, temp_data)
  }
  temp_obj_list <- temp_list[names(temp_list) %in% "object"]
  temp_ <- temp_obj_list[[1]]
  temp_obj <- as.data.frame(temp_[!names(temp_list$object) %in% c("part", "bndbox")])
  
  if (length(temp_obj_list) > 1){  
  for (i in c(2:length(temp_obj_list))){
    temp_ <- temp_obj_list[[i]]
    temp <- as.data.frame(temp_[!names(temp_) %in% c("part", "bndbox")])
    temp_obj <- bind_rows(temp_obj, temp)
  }
  }  
  temp_object <- as.data.frame(temp_obj)
  temp_fnl <- merge(data, temp_object)
}

# list of files
list_files <- list.files()
fnl_parsed_data <- get_temp(list_files[1])
for (i in c(2:length(list_files))){
  print(i)
  temp_data <- get_temp(list_files[i])
  fnl_parsed_data <- bind_rows(fnl_parsed_data, temp_data)
}
fnl_parsed_data$occluded[is.na(fnl_parsed_data$occluded)] <- 0
fnl_parsed_data$width <- as.integer(fnl_parsed_data$width)
fnl_parsed_data$height <- as.integer(fnl_parsed_data$height)
fnl_parsed_data$depth <- as.integer(fnl_parsed_data$depth)
fnl_parsed_data$segmented <- as.integer(as.character(fnl_parsed_data$segmented))
fnl_parsed_data$truncated <- as.integer(fnl_parsed_data$truncated)
fnl_parsed_data$difficult <- as.integer(fnl_parsed_data$difficult)
fnl_parsed_data$occluded <- as.integer(fnl_parsed_data$occluded)

setwd('/home/akash/github/face_segmentation/results')

####### False Positive #############################################################################
false_positive <- fread('false_positive_outputs.txt')
names(false_positive)[2] <- 'Class'
false_positive_v2 <- left_join(false_positive, class_mapping)
false_positive_v2$iou_k <- false_positive_v2$inter_k / false_positive_v2$union_k
sum(false_positive_v2$iou_k < 0.5) # 1229 / 1906
false_positive_v2$tag <- ifelse(false_positive_v2$iou_k < 0.5, 'false_positive', 'None') # 1229
false_positive_v2$exclude <- ifelse(false_positive_v2$pred_pixels_k < 50, 1, 0)
sum(false_positive_v2$exclude) # 147

false_positive_v3 <- false_positive_v2

false_positive_v3$tag <- ifelse(false_positive_v3$iou_k>0.1 & false_positive_v3$iou_k<0.5, 'LC', false_positive_v3$tag)
false_positive_v3$p_acc_sc <- false_positive_v3$inter_sc / false_positive_v3$pred_pixels_k
false_positive_v3$tag <- ifelse(false_positive_v3$tag == 'false_positive' & false_positive_v3$p_acc_sc > 0.1, 'SC', false_positive_v3$tag)

false_positive_v3$p_acc_dc <- false_positive_v3$inter_dc / false_positive_v3$pred_pixels_k
false_positive_v3$tag <- ifelse(false_positive_v3$tag == 'false_positive' & false_positive_v3$p_acc_dc > 0.1, 'DC', false_positive_v3$tag)

####### trying to get pixel_accuracy as also one of the class
false_positive_v3$p_acc_k <- false_positive_v3$inter_k / false_positive_v3$pred_pixels_k
false_positive_v3$tag_2 <- ifelse(false_positive_v3$p_acc_k < 0.5, 'fp', 'None')
false_positive_v3$tag_2 <- ifelse(false_positive_v3$p_acc_k< 0.5 & false_positive_v3$p_acc_k>0.1, 'LC', false_positive_v3$tag_2)
false_positive_v3$tag_2 <- ifelse(false_positive_v3$tag_2 == 'fp' & false_positive_v3$p_acc_sc > 0.1, 'SC', false_positive_v3$tag_2)
false_positive_v3$tag_2 <- ifelse(false_positive_v3$tag_2 == 'fp' & false_positive_v3$p_acc_dc > 0.1, 'DC', false_positive_v3$tag_2)

## Looking at effect of various tags
effect_of_tags_v1 <- false_positive_v3%>%
  group_by(name, tag_2) %>%
  summarise(inter_k = sum(inter_k), union_k = sum(union_k))

effect_of_tags_v1$sum_union_k <- sum(effect_of_tags_v1$union_k)
effect_of_tags_v1$effect_tag_2 <- (effect_of_tags_v1$union_k - effect_of_tags_v1$inter_k) / effect_of_tags_v1$sum_union_k

######## False Negative #############################################################################
false_negative <- fread('False_Negative_Database.txt')
class_mapping <- fread('class_mapping.txt')

data_at_class_lvl <- fnl_parsed_data %>%
  group_by(folder, filename, database, annotation, image, width, height, depth, segmented, name) %>%
  summarise(pose = Mode(pose), truncated = max(truncated), difficult=max(difficult), occluded=max(occluded))

false_negative_v2 <- left_join(false_negative, class_mapping)
data_at_class_lvl_v2 <- left_join(data_at_class_lvl, false_negative_v2, by=c("name", "filename"))
# write.csv(data_at_class_lvl_v2, "./false_negative.csv")
data_at_class_lvl_v2$class_size_pixel = data_at_class_lvl_v2$No_Pixel / (500*500)
class_size_tags <- data_at_class_lvl_v2 %>%
  group_by(name) %>%
  summarise(xs_size = quantile(class_size_pixel, probs=0.1), 
            s_size = quantile(class_size_pixel, probs=0.3),
            m_size = quantile(class_size_pixel, probs=0.7),
            l_size = quantile(class_size_pixel, probs=0.9))

false_negative_v3 <- left_join(data_at_class_lvl_v2, class_size_tags)
false_negative_v4 <- false_negative_v3 %>%
  mutate(class_size = ifelse(class_size_pixel < xs_size , 'XS', 
                             ifelse(class_size_pixel < s_size, 'S', 
                                    ifelse(class_size_pixel < m_size, 'M', 
                                           ifelse(class_size_pixel < l_size, 'L', 'XL')))))
false_negative_v4$mean_iou <- false_negative_v4$Intersection / false_negative_v4$Union
sum(false_negative_v4$mean_iou < 0.5) # 440
false_negative_v4$fn <- ifelse(false_negative_v4$mean_iou < 0.5, 1, 0)

# Occlusion effect
occlusion_effect = false_negative_v4 %>%
  group_by(name, occluded) %>%
  summarise(mean_iou = mean(mean_iou), nobs=n())

# Truncation effect
truncated_effect = false_negative_v4 %>%
  group_by(name, truncated) %>%
  summarise(mean_iou = mean(mean_iou), nobs=n())

# Class size effect
class_size_effect = false_negative_v4 %>%
  group_by(name, class_size) %>%
  summarise(mean_iou = mean(mean_iou), nobs=n())

# View point effect
vp_effect = false_negative_v4 %>%
  group_by(name, pose) %>%
  summarise(mean_iou = mean(mean_iou), nobs = n())

# View point effect
difficult_effect = false_negative_v4 %>%
  group_by(name, difficult) %>%
  summarise(mean_iou = mean(mean_iou), nobs = n())



mean_iou = fread('iou_outputs.txt')
new_file <- inner_join(fnl_parsed_data, mean_iou)

# Effect of occulusion
occlusion_output = new_file %>%
  group_by(occluded) %>%
  summarise(iou = mean(mean_iou, na.rm=T), nobs = n())

occlusion_output = new_file %>%
  group_by(occluded, name) %>%
  summarise(iou = mean(mean_iou, na.rm=T), nobs = n())

occlusion_output = new_file %>%
  group_by(occluded, name, pose) %>%
  summarise(iou = mean(mean_iou, na.rm=T), nobs = n())

# Truncated list
truncated_output = new_file %>%
  group_by(occluded) %>%
  summarise(iou = mean(mean_iou, na.rm=T), nobs = n())

truncated_output = new_file %>%
  group_by(occluded, name) %>%
  summarise(iou = mean(mean_iou, na.rm=T), nobs = n())

truncated_output = new_file %>%
  group_by(occluded, name, pose) %>%
  summarise(iou = mean(mean_iou, na.rm=T), nobs = n())

