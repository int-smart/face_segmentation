library(XML)
library(dplyr)
setwd('/home/akash/github/face_segmentation/data/VOC2012/Annotations//')

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

  for (i in c(2:length(temp_obj_list))){
    temp <- temp_obj_list[[i]][!names(temp_list$object) %in% c("part", "bndbox")]
    temp <- as.data.frame(temp)
    temp_obj <- bind_rows(temp_obj, temp)
  }
  temp_object <- as.data.frame(temp_obj)
  temp_fnl <- merge(data, temp_object)
}

# list of files
list_files <- list.files()
fnl_parsed_data <- get_temp(list_files[1])
for (i in c(2:length(list_files))){
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
