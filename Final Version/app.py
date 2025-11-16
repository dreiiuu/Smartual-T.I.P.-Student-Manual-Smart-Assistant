# ============================================================================
# SMART STUDENT ASSISTANT - T.I.P. Student Manual Q&A System
# A Streamlit application using semantic search and in-context classification
# Author: AI Assistant | Date: November 2025
# ============================================================================


import zipfile
import gdown
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime

# ============================================================================
# CONFIGURATION & COLOR PALETTE
# ============================================================================

# Download TIP LOGO from GitHub if not exists
if not os.path.exists("TIP LOGO.jpg"):
    try:
        logo_url = "https://raw.githubusercontent.com/dreiiuu/Smartual-T.I.P.-Student-Manual-Smart-Assistant/main/Final%20Version/TIP%20LOGO.jpg"
        response = requests.get(logo_url)
        if response.status_code == 200:
            with open("TIP LOGO.jpg", 'wb') as f:
                f.write(response.content)
            print("✅ TIP LOGO downloaded from GitHub")
        else:
            print("❌ Could not download TIP LOGO")
    except Exception as e:
        print(f"❌ Error downloading logo: {e}")


MANUAL_DATA_FILE = "manual_data.json"
SECTION_EXAMPLES_FILE = "section_examples.json"

manual_data = {
  "General Information": "\nT.I.P. General Information: The Technological Institute of the Philippines (T.I.P.) was established on February 8, 1962, \nby Engineer Demetrio A. Quirino, Jr. and Dr. Teresita U. Quirino as a private non-sectarian stock school in Manila.\n\nVision: We envision a better life for Filipinos by empowering our students with the best globally competitive technological \neducation in engineering, computing, and allied disciplines.\n\nMission: Through digitalization and innovation in academic design and delivery, T.I.P. students, faculty, staff and industry \npartners work together in both traditional and online/flexible learning to transform our students to achieve optimal students outcomes.\n\nCore Values: Commitment to Continuous Improvement and Innovation, Collaborative Mindset, Community Spirit, Service Orientedness, \nPositive Attitude for Learning and Working, Effective and Open Communication, Digitally Savvy.\n\nGraduate Attributes: Professional Competence, Communication Skills, Critical Thinking and Problem Solving Skills, \nSocial and Ethical Responsibility, Interpersonal Skills, Productivity, Lifelong Learning.\n\nProgram Offerings include Engineering and Architecture (BSArch, BSChE, BSCE, BSCpE, BSEE, BSECE, BSEnSE, BSIE, BSME), \nComputer Studies (BSCS, BSDSA, BSIT, BSIS, BSEMC), Business Education (BSA, BSAIS, BSBA), Teacher Education (BSEd, BSNEd, TCP), \nand Arts programs.\n\nAwards and Recognitions: T.I.P. Manila and T.I.P. Quezon City were awarded Autonomous Status by CHED in April 2016. \nThe institution has ABET accreditation, Seoul Accord recognition, and AUN-QA assessment for select programs.\n",
  "Admissions": "\nStudent Eligibility for Admissions: Students who satisfy any of the following may apply for admission to T.I.P.: \n1) Graduates of secondary education recognized by DepEd and not enrolled in any tertiary program, \n2) Passers of PEPT or ALS following DepEd regulations, 3) College Transferees, 4) Second Degree Applicants, 5) Cross-Enrollees.\n\nAdmission Requirements for First Year Filipino Students: Original copy of Senior High School Report Card (Form 138/SF9) from Grade 12, \nOriginal copy of PSA Birth Certificate, Certificate of Good Moral Character (with school seal), Two 2\"x2\" recent ID pictures, \nCertificate of Honors/Rank if applicable, Mandatory drug test.\n\nFor Transferees/Second Degree Applicants: Original copy of Transfer Credentials, Transcript of Records or True Copy of Grades \nfrom last school attended, Certificate of Good Moral Character, Two 2\"x2\" recent ID pictures, Mandatory drug test.\n\nFor International Students: A separate set of guidelines shall apply for admission of international students.\n\nEnrollment Procedure: A student applicant who has complied with all admission requirements is qualified to enroll. \nIf requirements (except Form 138/SF9/Transfer Credential/ALS/PEPT result) are not available, the applicant must execute \nan UNDERTAKING and comply within the term of first enrollment.\n",
  "Registration and Enrollment": "\nEnrollment Procedure and Regulations: A student shall be considered officially enrolled if one has submitted all required \nadmission credentials and has paid in full, or made arrangement to pay the tuition and other fees for the semester on installment basis.\n\nCourse Load: The course load and sequence of courses shall be in accordance with the approved curriculum for each program of study. \nNo course may be taken unless the prerequisite courses have been taken and passed. Otherwise, the registration shall be invalidated.\n\nFor Engineering and Computing programs, the Design/Capstone Course shall be enrolled ONLY in the terminal/last term of the student.\n\nA grade of 5.00 (Failed) in any prerequisite course disqualifies a student from enrolling in the next-higher course.\n\nCancellation of Enrollment/Withdrawal: A student who wishes to withdraw studies before start of classes or within first two weeks \nmust notify the Registrar in writing through the Program Chair/College Dean, and copy furnish Student Accounting Services.\n\nProgram Shifting: A student shall be allowed to shift to another program provided one complies with program residency requirement \nof one (1) school year attendance in T.I.P. or at least forty-two (42) units. Program shifting may be allowed only once.\n\nCross Enrollment: Cross-enrollment is only permitted if the total study load does not exceed the regular study load for the semester. \nT.I.P. students are not allowed to cross-enroll in more than one (1) school in a semester.\n",
  "Grading System": "\nGrading System: There are three major examinations to complete a course: prelim, midterm, and final exams.\n\nGrade Scale: 1.00 (94-100) Excellent, 1.25 (88.5-93.99) Superior, 1.50 (83-88.49) Meritorious, 1.75 (77.5-82.99) Very Good, \n2.00 (72-77.49) Good, 2.25 (66.5-71.99) Very Satisfactory, 2.50 (61-66.49) Satisfactory, 2.75 (55.5-60.99) Fair, \n3.00 (50-55.49) Passing, 5.00 (0-49.99) Failed, 4.00 Incomplete, 6.00 Official Dropped, 7.00 Unofficially Dropped, \n8.00 No Credit, 9.00 Withdrawn.\n\nGrade Computation Formulas:\nPG = 0.50 PE + 0.50 CSP (Prelim Grade)\nMG = 1/3 PG + 2/3 (0.50 ME + 0.50 CSM) (Midterm Grade)\nFG = 1/3 MG + 2/3 (0.50 FE + 0.50 CSF) (Final Grade)\n\nWhere PE = Prelim Exam, ME = Midterm Exam, FE = Final Exam, CS = Class Standing\n\nA grade of 4.0 (Incomplete): Failure to remove within one-year prescribed period shall automatically result in grade of 5.0 (Failed). \nThe student is required to re-enroll in the course.\n\nA grade of \"NC\" or No Credit shall be given to a student who did not take the final examination or did not submit an academic \nrequirement for completion of a course and the student's scholastic performance is not sufficient to merit a passing grade.\n",
  "Academic Probation and Retention": "\nStudent Retention: A full-time student who failed in more than 50% of one's registered units, including PE and CWTS, \nshall be placed on academic probation. The school shall allow three (3) probationary terms for the duration of student's program of study.\n\nFor students on academic probation in programs NOT requiring licensure examination: allowed to re-enroll on reduced load - \nFirst Probation: 3 units less from previous semester or 9 units (whichever higher); \nSecond Probation: Same as first probation; Third Probation: Same rules apply.\n\nFor students on academic probation in programs REQUIRING licensure examination: \nFirst Probation: 3 units less from previous semester or 9 units (whichever higher); \nSecond Probation: 3 units less from previous semester or 9 units (whichever higher) AND/OR student shall be advised \nto shift to program not requiring board examination; Third Probation: Similar restrictions.\n\nA student under Third Probation status who fails in one course shall no longer be readmitted in succeeding term.\n\nFor transferees: A transferee shall be placed under Academic Probation if one has failed and/or dropped more than 50% \nof units enrolled during previous semester. The transferee shall enroll on reduced load of eighteen (18) units and \nattend regular counseling session during the semester.\n",
  "Graduation Requirements": "\nGraduation: A student who has successfully completed all courses in one's curriculum, complied with all graduation requirements, \nand fulfilled residency requirement of at least one (1) school year attendance in T.I.P. or at least forty-two (42) units is \neligible for graduation.\n\nDeemed ineligible for graduation: student with pending grade/s of 4.00 (Incomplete); 5.00 (Failed); 6.00 (Officially Dropped); \n7.0 (Unofficially Dropped); unsettled financial and property obligations with the school; unresolved disciplinary cases; \npenalty of exclusion/expulsion.\n\nGraduation Honors - Latin Honors awarded only to graduating students with four and/or five-year degree program:\n- Summa Cum Laude: GPA 1.00 to 1.25 with no grade in any course below 1.50\n- Magna Cum Laude: GPA 1.26 to 1.50 with no grade in any course below 2.00\n- Cum Laude: GPA 1.51 to 1.75 with no grade in any course below 2.50\n\nTo qualify for Latin Honors, graduating student must have earned at least 75% of required courses of the program at T.I.P. \nNon-academic grades (e.g., NSTP) are not included in GPA computation for Latin Honor.\n\n\"With Distinction\" award given to graduating student who may satisfy GPA requirement for Latin Honor but obtains low grades \n(below 2.5) in any course, including PE and NSTP.\n\nAny violation of school rules and regulations punishable with suspension shall constitute grounds for disqualification from \nreceiving any and all academic honors from T.I.P.\n",
  "Scholarships and Financial Aid": "\nScholarships: T.I.P. is committed to bring the opportunity of higher education to students who earnestly desire it and \ndemonstrate the ability to benefit from it. The school offers various scholarship programs.\n\nMajor scholarship grants: \n1) Engr. Demetrio A. Quirino Jr. Scholarships for Engineering and Architecture\n2) Dr. Teresita U. Quirino Scholarships for Accountancy and Teacher Education\n\nScholarship Monitoring: For scholars, a scholarship coordinator is tasked to monitor their performance regularly and help them \naddress academic difficulties. Orientation of new scholars is conducted to facilitate immersion and mentoring. \nParticipation of scholars in exchange programs and competitions inside and outside the school is encouraged.\n\nThe scholarship coordinator facilitates the organization and operation of the scholars' association to promote their welfare.\n",
  "Student Conduct and Discipline": "\nNorms of Conduct for Students: All students are obliged to read announcements on Canvas LMS, website, official T.I.P. social media accounts, \nand bulletin boards to keep abreast of T.I.P. activities. Students have rights to express concerns, be represented through Student Council, \nform organizations, and defend oneself against charges.\n\nID Card Policy: For proper identification, student must wear ID card upon entering campus. No ID, no entry. \nAnyone caught without ID while inside campus shall be subject to disciplinary sanctions.\n\nAttire and Grooming: The approved school uniform for male and female students shall be worn inside campus during school days \nexcept non-school uniform day/s. For online classes, dress code prescribed in T.I.P. Policy on Student Protocols and Etiquette \nfor Online Learning shall be observed.\n\nFor male students NOT allowed: long hair (should be tidy, combed, properly trimmed), hair accessories, multi-colored/loud hair coloring, \ncolored nail polish, ear/nose/tongue/other visible piercings, visible tattoos, shorts, sandals/slippers, caps/hats, scarves.\n\nFor female students NOT allowed: more than one earring per ear, ear/nose/tongue/other visible piercings, visible tattoos, \nmulti-colored/loud hair coloring.\n\nClassroom Behavior: Student should abide by usual classroom policies and procedures implemented by teachers, both in-person and virtual. \nTeacher responsible to report to OSA any student violating these policies.\n\n5S Japanese Model: Every student expected to observe proper norms of politeness, etiquette, and courtesy. \n5S represents: Seiri (Sort/Suriin), Seiton (Systemize/Sinupin), Seiso (Sweep/Simutin), Seiketsu (Sanitize/Siguruhin ang Kallinisan), \nShitsuke (Self-discipline/Sariling Kusa).\n\nDecorum: Faculty members, administrators, or staff have special parental authority (loco parentis) for students. \nThey are obligated to call attention of students who display unbecoming behavior inside or outside campus.\n",
  "Student Organizations": "\nStudent Organizations: Membership in any recognized student organization is voluntary. However, recruitment of first-year students \nto fraternities/sororities is strictly prohibited. Students should immediately report incidents of forced recruitment and other \nfraternity-related cases such as hazing, threat, intimidation, coercion, extortion to SOHAS, GCC, or OSA.\n\nUse of Campus Facilities: School allows students to use campus facilities (PE Center, Big Court, Seminar Rooms, Congregating Area, \nStudy Areas) for programs, meetings, recreational and educational activities. School has right to deny use of facilities to those \nunable or unwilling to abide by school rules and regulations.\n\nRepresentation: Student may not participate in any outside activity, contest, conference, field trip, association, society, or group \nas representative of school or any recognized student organizations without written authorization from OSA, VPSAS, VPAA, or T.I.P. President.\n\nSchool shall not be held responsible for consequences resulting from student's/student organization's participation in outside activity \ndone in individual's/organization's personal capacity without school's prior written approval.\n\nAnnouncements and Publications: Student must obtain clearance from OSA before posting school-related materials/announcements. \nMaterials carrying school's name, logo, identification (print or digital, inside/outside campus) must be submitted for vetting \nto both OSA and Marketing and Communications Office (MCO). This applies to announcements in newspapers/magazines, posters, \nstreamers in public spaces, and social media profiles of student councils and recognized organizations.\n",
  "Attendance Policy": "\nAttendance: Prompt and regular attendance in all classes required of all students from very first meeting of every course.\n\nStudent who incurs absences of more than twenty (20%) percent of prescribed number of class or laboratory periods during term \nshall not earn credit for the course.\n\nStudent considered officially enrolled only if: submitted appropriate admission/transfer credentials; paid school fees (down payment) \nupon enrollment; been authorized to attend classes by school.\n\nExcuse Letters: In case of absence, student must present letter of excuse to all teachers concerned. If student is minor or \neven if turned 18 but still wholly dependent upon parents for support, letter must be signed by parent or guardian.\n\nMedical Absences: In case of absence due to illness, excuse letter or medical certificate must be validated by Medical and Dental Services (MDS).\n\nStudent held responsible for all lessons and assignments missed during absence.\n\nApproved Absence: applies to student authorized in writing by OSA or other school officials to represent T.I.P. in off-campus \nfunction/activity or attend meeting with school officials.\n\nTardiness Policy: Student considered late/tardy if arrival in class within: 1-hour period - 15 minutes; 1.5-hour period - 25 minutes; \n2-hour period - 30 minutes; 3-hour period - 45 minutes. Student tardy three (3) times considered as absent for one (1) meeting.\n\nDropped Status: Student who stopped attending classes and/or incurred absences equivalent to more than 20% of school days \nwith or without notification shall not be given credit. Marks: \"6.00\" if 20% limit exceeded before Midterm with notification; \n\"7.00\" if exceeded without notification; \"5.00\" if exceeded after Midterm with failing Midterm Grade.\n",
  "Student Services": "\nStudent Services Available at T.I.P.:\n\n1. Career Services: Career Centers provide high-quality services to support students at all stages including career training, \ndevelopment, advising, and services helping students realize value of their degree.\n\n2. Library: Practices combination of open and semi-closed shelf systems. Services include: Orientation and Instruction, \nReference and Information, Internet Services, OPAC, Bibliographic Services, Current Awareness, Instructional Media, \nPhotocopying and Printing, Inter-referral, Wi-fi. Also provides Virtual Reference Assistance, Online Document Delivery, \nOnline Bibliographic Service, Online Library Orientation, Online Learning Resources.\n\n3. Medical and Dental Services (MDS): Clinic provides consultations, treatment for common medical/dental illnesses, \nbasic first aid treatment. Laboratory workups and special diagnostic procedures issued as necessary. Dental services include \noral examination, oral prophylaxis, temporary filling, common dental treatments and emergencies. Online Medical/Dental Consultation \nintegrated into Student Well Being Program (SWBP) accessible via T.I.P. Canvas LMS.\n\n4. Office of Student Affairs (OSA): Committed to encourage students to experience meaningful and excellent academic life in \nsafe and harmonious environment. Three main services: Student Leadership Development, Student Discipline, and Institutional Student Support Programs.\n\n5. Guidance and Counseling Center (GCC): Provides variety of services responsive to needs of individual students or special groups. \nOffers: Counseling Service, Information Service, Individual Inventory, Testing Service, and Referral Service.\n\n6. Math Enhancement Program (MEP): Free Summer Math Tutorial for incoming Engineering, Architecture, and IT first-year students \nto prepare them to take college-level math courses. Special intervention program in Mathematics designed to identify special learning needs.\n\n7. Online Study Group Tutorial (OSGT): Free tutorial in Mathematics and English for students who need extra help. \nObjectives: clarification of earlier lessons; reinforce key concepts; serve as advanced coursework; serve as review for board examinations. \nSessions facilitated by faculty members and pre-selected students as peer tutors/teaching assistants.\n\n8. Student Advising: Advising system provides individual advice and assistance in completion of program of study. \nGuides students about academic policies, career options, instructional support, job opportunities. Curriculum advising involves \nassisting students in selecting courses each term. Career advising covers employment opportunities, clarifications about sub-fields, \npossibilities of graduate studies, career-related concerns.\n\n9. Security, Occupational Health, and Safety Office (SOHAS): Security and safety is primary concern. Friendly and reliable security guards \nroaming campus as scheduled. Adequate CCTV cameras installed for surveillance. Strong partnership with policemen and barangay personnel. \nGuided by Emergency Management Manual with response protocols and countermeasures to various emergencies.\n\n10. T.I.P. EXCEL: Student success program aims to prepare T.I.Pians to accomplish current and future academic, personal, \nand professional goals through value-adding programs. Integrates: T.I.P. LEADS (alternative-design systems), T.I.P. ENGAGE \n(high student retention), T.I.P. Student Well-being Program (online services), T.I.P. CONNECT (student concerns).\n",
  "Tuition and Fees": "\nPayment of Tuition and Other School Fees: Tuition and other school fees shall be payable in cash or installment payments \nin accordance with schedule provided by Student Accounting Services.\n\nNotice of any increase in fees shall be announced in advance through posting on bulletin boards in conspicuous places in school premises.\n\nUpon enrollment, student and/or parent/s/guardian/s bind themselves to pay corresponding tuition and other school fees whether \nstudent completes or does not complete studies during given semester.\n\nStudent and parent/s/guardian/s agree they are bound to recognize policy that student shall be readmitted only if tuition and \nother school fees are paid in full.\n\nRefund Policy: Student who files for cancellation/withdrawal/discontinuance of studies within first two (2) weeks of classes \nduring regular semester or within first four (4) days of classes during summer and has paid pertinent fees may apply for refund:\n\n- Administrative/processing fee if one withdraws before start of classes\n- Ten percent (10%) of total amount due if withdraws within first week of classes in regular semester or within two (2) days in summer\n- Twenty percent (20%) of total amount due if withdraws within second week of classes in regular semester or on third/fourth day in summer\n- Full fees due for term if withdraws any time after second week of classes in regular semester or after fourth day of classes in summer\n\nProcessing is one (1) month after start of classes. Follow-ups for refunds may be done only a month after start of classes.\n\nUndersized Classes: Full class defined as class consisting of fifteen (15) or more students. Undersized class has fewer than fifteen (15) students.\n\nClass of 6-14 students may be retained if: students pay for shortfall (difference between what 15-student class would have paid and \nwhat undersized class actually paid); OR assigned faculty member requests authority to handle class at modified rate (80% of faculty rate \nor 80% of tuition paid by undersized class, whichever lower).\n\nClass of 5 students or less may be retained if students shoulder shortfall. Otherwise, class shall be dissolved.\n",
  "Disciplinary Offenses": "\nClassification of Offenses and Sanctions:\n\nMinor Offenses include:\n1. Failure to abide with health and safety protocols\n2. Non-wearing of school ID card\n3. Tampered/unvalidated school ID card\n4. Improper haircut/style, improper hair dye color\n5. Improper or non-wearing of school uniform\n6. Non-observance of dress code\n7. Wearing inappropriate jewelry/accessories\n8. Cross-dressing\n9. Refusal to submit to lawful inspection/search\n10. Failure to return library books on time\n11. Lost/damaged library books\n12. Use of mobile phones during class hours\n13. Unauthorized posting of announcements\n14. Running, shouting, whistling\n15. Sitting on armchairs/tables causing damage\n16. Littering or spitting\n17. Loitering in corridors\n18. Chewing gum\n19. Hiding valuable property of others\n20. Public display of affection\n\nMajor Offenses - Academic Dishonesty:\n1. Forgery or falsification and/or alteration or misrepresentation of academic/official school records or school-related documents\n2. All forms of cheating in any examination, test, quiz, project, report, or assignment\n3. Academic dishonesty such as plagiarism; passing off someone else's work as one's own\n\nGeneral Rules of Conduct and Discipline: All cases involving discipline of students shall be subjected to jurisdiction of \nOffice of Student Affairs (OSA). OSA shall conduct investigation, act upon minor offenses, issue preventive suspension orders \npending investigation when necessary, and for major offenses recommend to President creation of investigating committee.\n\nFor violations that are criminal in nature, school reserves right to turn over student to police authorities. \nAdministrative charges may also be filed without prejudice to existing penal laws.\n",
  "Educational Philosophy": "\nT.I.P.'s Educational Philosophy: T.I.P. believes that when students are immersed in constructivist experiential technological \nteaching and learning focused on outcomes, and imbued with grit, determination and love for fellow Filipino, they become lifelong learners, \ninnovators, and problem-solvers for the nation.\n\nOpen-Door Policy: T.I.P. is committed to bring opportunity of higher education to students who earnestly desire it and demonstrate \nability to benefit from it. Consistent with philosophy of making higher education accessible to many, T.I.P. offers various scholarship programs.\n\nOutcomes-Based Education (OBE): In continuing quest for excellence and spirit of continuous quality improvement, T.I.P. implemented \nOutcomes-Based Education as strategy to achieve long-term objectives for graduates. OBE is working backward with students as center \nof learning-teaching milieu.\n\nImplementation driven by: 1) regulatory bodies (CHED, PRC, IMO); 2) local and international accrediting bodies (PACUCOA, PTC-ACBET-EAC, ABET); \n3) international certifying bodies; 4) feedback from external stakeholders.\n\nGuided by internal policies: 1) Vision, Mission, Core Values, Core Competencies; 2) Quality Policy; 3) T.I.P. initiatives supporting OBE \n(Faculty and Staff Development Program, Student Development Program).\n\nOutcomes-Based Teaching and Learning (OBTL): T.I.P. embarked on proactive plan to implement OBTL in all academic programs using \nCity University of Hong Kong OBTL model. Framework revolves around: 1) Intended Learning Outcomes (ILOs), 2) Teaching and Learning Activities (TLAs), \n3) Assessment Tasks (ATs).\n\nOBTL is student-centered approach to delivery of educational programs where curriculum topics expressed as intended outcomes for students to learn. \nTeachers facilitate, students actively engaged in learning process. About re-aligning intended learning outcomes with teaching and assessment, \nfocusing on what graduates know, what they can do, and their personal attributes.\n\nStudent Development Program (SDP): Aims to produce graduates with full competence in fields of study and possess Filipino industry-desired \nand global citizen values to enhance employability and desirability by industry after graduation. SDP mainstreamed in identified courses. \nConsists of modules: Self-Awareness, Goal Setting, Values Development, Developing Winner's Mindset. SDP serves as vehicle of OBTL addressing \nspecific components of program educational objectives and student outcomes.\n"
}


section_examples = {
  "General Information": [
    "What is the vision of T.I.P.?",
    "When was T.I.P. established?",
    "What programs does T.I.P. offer?",
    "What are the core values of T.I.P.?",
    "Tell me about T.I.P.'s accreditations",
    "What is the mission of the school?"
  ],
  "Admissions": [
    "How do I apply to T.I.P.?",
    "What are the admission requirements?",
    "What documents do I need for admission?",
    "Can international students apply?",
    "What is required for transferees?",
    "Do I need to take an entrance exam?"
  ],
  "Registration and Enrollment": [
    "How do I enroll in courses?",
    "Can I shift to another program?",
    "What are the rules for cross enrollment?",
    "How do I withdraw from a course?",
    "What is program shifting?",
    "Can I change my program?"
  ],
  "Grading System": [
    "How are grades calculated at T.I.P.?",
    "What does a grade of 1.00 mean?",
    "How is the final grade computed?",
    "What happens if I get an incomplete grade?",
    "What is the passing grade?",
    "Explain the grading scale"
  ],
  "Academic Probation and Retention": [
    "What is academic probation?",
    "What happens if I fail most of my courses?",
    "How many units can I take on probation?",
    "Can I be dismissed for poor grades?",
    "What are the probation rules?",
    "How many probationary terms are allowed?"
  ],
  "Graduation Requirements": [
    "What are the requirements to graduate?",
    "How can I graduate with honors?",
    "What is Summa Cum Laude?",
    "What GPA do I need for Latin honors?",
    "How many units do I need to graduate?",
    "What disqualifies me from graduation?"
  ],
  "Scholarships and Financial Aid": [
    "What scholarships are available?",
    "How can I apply for financial aid?",
    "Are there scholarships for engineering students?",
    "What is the Quirino scholarship?",
    "How do I maintain my scholarship?",
    "Tell me about scholarship programs"
  ],
  "Student Conduct and Discipline": [
    "What are the dress code requirements?",
    "Can I wear civilian clothes?",
    "What is the ID policy?",
    "What hairstyles are allowed?",
    "Are tattoos allowed?",
    "What is the 5S program?"
  ],
  "Student Organizations": [
    "Can I join fraternities?",
    "How do I start a student organization?",
    "What facilities can student orgs use?",
    "Are freshmen allowed in fraternities?",
    "How do I get approval for student events?",
    "What are the rules for student activities?"
  ],
  "Attendance Policy": [
    "What is the attendance requirement?",
    "How many absences are allowed?",
    "What happens if I miss too many classes?",
    "How is tardiness counted?",
    "Do I need an excuse letter for absences?",
    "What is the 20% attendance rule?"
  ],
  "Student Services": [
    "What services are available for students?",
    "Where is the library?",
    "Does T.I.P. have a clinic?",
    "What is the guidance center?",
    "Is there tutoring available?",
    "Tell me about career services"
  ],
  "Tuition and Fees": [
    "How much is the tuition?",
    "Can I pay in installments?",
    "What is the refund policy?",
    "How do I pay my fees?",
    "What happens if I withdraw from classes?",
    "Are there fees for late enrollment?"
  ],
  "Disciplinary Offenses": [
    "What are minor offenses?",
    "What happens if I cheat?",
    "What is considered academic dishonesty?",
    "Can I be suspended?",
    "What are the penalties for violations?",
    "What happens if I plagiarize?"
  ],
  "Educational Philosophy": [
    "What is outcomes-based education?",
    "What is T.I.P.'s teaching philosophy?",
    "What is OBE?",
    "What is the student development program?",
    "Tell me about OBTL",
    "What is the open door policy?"
  ]
}



# Save the data to files
with open(MANUAL_DATA_FILE, 'w', encoding='utf-8') as f:
    json.dump(manual_data, f, indent=2, ensure_ascii=False)

with open(SECTION_EXAMPLES_FILE, 'w', encoding='utf-8') as f:
    json.dump(section_examples, f, indent=2, ensure_ascii=False)


FEEDBACK_PATH = "feedback_log.csv"
SCHOOL_LOGO = "tip_logo.png"
CHUNK_SIZE = 300


# ============================================================================
# DOWNLOAD MODEL FILES INDIVIDUALLY
# ============================================================================

os.makedirs("smartual_model", exist_ok=True)
os.makedirs("smartual_model/1_Pooling", exist_ok=True)

model_files = {
    "config_sentence_transformers.json": "1-dETJOjgjUzGBvaa2YR3JB6H-9lorUZp",
    "config.json": "1ZOOhmA-zPPnG0NkRxRTBb8knpkUpfj3B",
    "model.safetensors" : "1wjVbH3jEF4XwkKkQfvJaB7Wi5E1DV1MU",
    "modules.json" : "1YRcFvdaYhi8iJW3b_9_BXMNSln9nfZBJ",
    "sentence_bert_config.json" : "16nZlJM5pHvzxd5nZXNk0Qnmp2nmHGKTp",
    "vocab.txt": "1GXsT9SP16r65ycf7Q8-M744JfbZRqJjs",
    "tokenizer.json": "1y7W_4g6LjeMgycWE8EEMLzZ_keF5pSTB",
    "special_tokens_map.json": "1OxCv4kV7P5RIfyIZl4X1MtOiERjpwvCJ",
    "tokenizer_config.json": "1MW4oL-rJLzCcM4A8sp1d3JOTF1WK03ue",
    "README.md": "1iuxiZmhPWHRxup8QIiZKk5jaMJJtwL02",
}

pooling_files = {
    "config.json": "1qwM2zvZZ__bwtTZr84zUrWqQeIPDWL9q",  
}

print("📥 Downloading model files...")
downloaded_files = []
for filename, file_id in model_files.items():
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = f"smartual_model/{filename}"
        
        if not os.path.exists(output_path):
            gdown.download(url, output_path, quiet=False)
            print(f"✅ Downloaded: {filename}")
        else:
            print(f"✅ Already exists: {filename}")
        downloaded_files.append(filename)
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")

# Download pooling folder files - FIX THE PATH
print("📥 Downloading pooling files...")
for filename, file_id in pooling_files.items():
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        # FIXED: Changed from "pooling" to "1_Pooling"
        output_path = f"smartual_model/1_Pooling/{filename}"
        
        if not os.path.exists(output_path):
            gdown.download(url, output_path, quiet=False)
            print(f"✅ Downloaded: 1_Pooling/{filename}")
        else:
            print(f"✅ Already exists: 1_Pooling/{filename}")
        downloaded_files.append(f"1_Pooling/{filename}")
    except Exception as e:
        print(f"❌ Failed to download 1_Pooling/{filename}: {e}")

# Check if essential files were downloaded
essential_files = ["config.json", "vocab.txt"]
missing_essential = [f for f in essential_files if f not in downloaded_files]

if missing_essential:
    st.error(f"❌ Missing essential model files: {missing_essential}")
    st.stop()
else:
    print("🎯 All essential model files downloaded successfully!")

MODEL_PATH = "smartual_model"
print(f"🎯 Using model path: {MODEL_PATH}")


# COLOR PALETTE - Balanced Yellow
PRIMARY = "#FFA000"      # Perfect Amber Balance
SECONDARY = "#5D4037"    # Rich Brown
ACCENT = "#FFF8E1"       # Soft Yellow
BACKGROUND = "#FFFFFF"    # Clean White
TEXT = "#37474F"         # Readable Dark Gray
SUCCESS = "#4CAF50"      # Positive Green
WARNING = "#FF6D00"      # Attention Orange

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

@st.cache_data
def load_manual_from_json():
    """Load the pre-structured T.I.P. Student Manual data from JSON file."""
    if not os.path.exists(MANUAL_DATA_FILE):
        st.error(f"Manual data file '{MANUAL_DATA_FILE}' not found!")
        return {}, []
    
    with open(MANUAL_DATA_FILE, 'r', encoding='utf-8') as f:
        manual_sections = json.load(f)
    
    # Create chunks from each section
    chunks = []
    for section_name, section_text in manual_sections.items():
        sentences = [s.strip() for s in section_text.split('. ') if s.strip()]
        
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            if current_word_count + len(words) <= CHUNK_SIZE:
                current_chunk.append(sentence)
                current_word_count += len(words)
            else:
                if current_chunk:
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunks.append({
                        "section": section_name,
                        "chunk_text": chunk_text,
                        "section_text": section_text,
                    })
                current_chunk = [sentence]
                current_word_count = len(words)
        
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                "section": section_name,
                "chunk_text": chunk_text,
                "section_text": section_text,
            })
    
    all_sections = list(manual_sections.keys())
    return chunks, all_sections

@st.cache_data
def load_section_examples():
    """Load example questions for in-context classification."""
    if not os.path.exists(SECTION_EXAMPLES_FILE):
        st.warning(f"Section examples file '{SECTION_EXAMPLES_FILE}' not found!")
        return {}
    
    with open(SECTION_EXAMPLES_FILE, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    return examples

@st.cache_resource
def load_model():
    """Load the sentence transformer model - FIXED VERSION"""
    try:
        # FIRST try to load your custom model
        print(f"🔄 Attempting to load custom model from: {MODEL_PATH}")
        model = SentenceTransformer(MODEL_PATH)
        print("✅ Loaded custom model successfully!")
        
        # Test the model to ensure it works
        test_embedding = model.encode(["test sentence"], show_progress_bar=False)
        print(f"✅ Model test passed. Embedding dimension: {test_embedding.shape[1]}")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to load custom model: {e}")
        st.warning("⚠️ Using fallback model instead of custom model")
        
        # Fallback: try to use the Hugging Face model
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("✅ Loaded fallback model: all-MiniLM-L6-v2")
            return model
        except Exception as e2:
            st.error(f"❌ Failed to load fallback model: {e2}")
            st.stop()
            
@st.cache_resource
def build_index(_chunks, _model):
    """Create FAISS index for all chunks and save embeddings."""
    texts = [chunk["chunk_text"] for chunk in _chunks]
    chunk_embeddings = np.array(_model.encode(texts, show_progress_bar=False)).astype("float32")
    
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    
    return index, chunk_embeddings

def classify_question(question, model, section_examples):
    """Use in-context examples to classify question's section by similarity."""
    question_embed = model.encode([question])[0]
    best_section = None
    best_score = -1
    
    for section, examples in section_examples.items():
        example_embeds = model.encode(examples, show_progress_bar=False)
        sim = cosine_similarity([question_embed], example_embeds).mean()
        if sim > best_score:
            best_section = section
            best_score = sim
    
    return best_section, float(best_score)

def retrieve_chunks(question, model, chunks, index, chunk_embeddings, top_k=3):
    """Retrieve the top K most similar chunks using FAISS."""
    question_embed = np.array(model.encode([question], show_progress_bar=False)).astype("float32")
    _, I = index.search(question_embed, top_k)
    
    top_chunks = [chunks[i] for i in I[0]]
    similarities = cosine_similarity(question_embed, chunk_embeddings[I[0]]).flatten()
    
    return top_chunks, similarities

def generate_answer(question, top_chunk, model):
    """Extract 2-3 most relevant sentences from top chunk."""
    sentences = [s.strip() for s in top_chunk["chunk_text"].split('. ') if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        return top_chunk["chunk_text"][:200] + "...", 0.5
    
    sent_embeds = model.encode(sentences, show_progress_bar=False)
    q_embed = model.encode([question], show_progress_bar=False)
    
    sims = cosine_similarity(q_embed, sent_embeds).flatten()
    top_idx = sims.argsort()[-3:][::-1]
    key_sentences = [sentences[i] for i in top_idx]
    
    answer = '. '.join(key_sentences) + '.'
    confidence = float(sims[top_idx[0]]) if len(top_idx) > 0 else 0.5
    
    return answer, confidence

def save_feedback(question, answer, section, confidence, helpful):
    """Append user feedback to a CSV file."""
    feedback = {
        "timestamp": pd.Timestamp.now(),
        "question": question,
        "answer": answer,
        "section": section,
        "confidence": round(confidence, 3),
        "helpful": helpful,
    }
    df = pd.DataFrame([feedback])
    
    if not os.path.exists(FEEDBACK_PATH):
        df.to_csv(FEEDBACK_PATH, index=False)
    else:
        df.to_csv(FEEDBACK_PATH, mode='a', header=False, index=False)

def count_sections_from_feedback():
    """Count section frequency for analytics."""
    if not os.path.exists(FEEDBACK_PATH):
        return {}
    try:
        df = pd.read_csv(FEEDBACK_PATH)
        return df["section"].value_counts().to_dict()
    except:
        return {}
# ============================================================================
# STREAMLIT UI - MODERN DESIGN WITH CENTRALIZED COLORS
# ============================================================================

def setup_css():
    """Setup CSS with centralized color palette"""
    st.markdown(f"""
    <style>
    /* Main styles */
    .main-container {{
        background-color: {BACKGROUND};
        color: {TEXT};
    }}
    
    .header-section {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
    }}
    
    .main-title {{
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    
    .subtitle {{
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        font-weight: 400;
    }}
    
    /* Cards */
    .answer-card {{
        background: {BACKGROUND};
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .answer-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }}
    
    .metric-card:hover {{
        transform: scale(1.05);
    }}
    
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%);
        color: {BACKGROUND};
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 12px {PRIMARY}33;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, {WARNING} 0%, #FF8C00 100%);
        color: {BACKGROUND};
        box-shadow: 0 6px 20px {WARNING}66;
        transform: translateY(-2px);
    }}
    
    .secondary-button {{
        background: linear-gradient(135deg, {SECONDARY} 0%, #616161 100%) !important;
        color: {BACKGROUND} !important;
    }}
    
    /* Sidebar */
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid #dee2e6;
    }}
    
    /* Input fields */
    .stTextInput>div>div>input {{
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem;
        font-size: 1rem;
        background-color: {BACKGROUND};
        color: {TEXT};
    }}
    
    .stTextInput>div>div>input:focus {{
        border-color: {PRIMARY};
        box-shadow: 0 0 0 2px {PRIMARY}33;
    }}
    
    /* Expander */
    .stExpander {{
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        background-color: {BACKGROUND};
    }}
    
    /* Progress bar */
    .stProgress > div > div > div {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%);
    }}
    
    /* Sample question buttons */
    .sample-question {{
        background: {BACKGROUND};
        border: 2px solid {PRIMARY};
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: {TEXT};
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        text-align: left;
    }}
    
    .sample-question:hover {{
        background: {PRIMARY};
        color: {BACKGROUND};
        transform: translateX(5px);
    }}
    
    /* Feedback buttons */
    .feedback-btn {{
        margin: 0.5rem;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    /* Custom sections */
    .welcome-section {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%);
        color: {BACKGROUND};
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
    }}
    
    .search-section {{
        background: {BACKGROUND};
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin: 2rem 0;
        border: 1px solid #e0e0e0;
    }}
    
    /* Answer highlight */
    .answer-highlight {{
        background: {ACCENT}33;
        border-left: 4px solid {PRIMARY};
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }}
    
    /* Section colors */
    .section-badge {{
        background: {PRIMARY};
        color: {BACKGROUND};
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Smart Student Assistant - T.I.P.",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Setup CSS with centralized colors
    setup_css()
    
    # ========================================================================
    # LOAD RESOURCES - WITH VALIDATION
    # ========================================================================
    
    # Show loading message
    with st.spinner("🔄 Loading AI model and resources..."):
        model = load_model()
        chunks, all_sections = load_manual_from_json()
        section_examples = load_section_examples()

    
    # Display which model is being used
    #st.sidebar.markdown("---")
    #if "custom" in MODEL_PATH.lower():
     #   st.sidebar.success("✅ Using Custom Model")
    #else:
     #   st.sidebar.success("⚠️ Using Fallback Model")
    
    #if not chunks:
     #   st.error("⚠️ Failed to load manual data. Please ensure 'manual_data.json' exists.")
      #  return
    
    
    index, chunk_embeds = build_index(chunks, model)
    
    # ========================================================================
    # SIDEBAR - MODERN DESIGN
    # ========================================================================
    
    with st.sidebar:
        # School Logo Section
        try:
            st.image("TIP LOGO.jpg", use_container_width=True)
        except:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%); 
                        border-radius: 15px; margin-bottom: 2rem; color: white;'>
                <h2 style='margin: 0; font-size: 2rem;'>🎓</h2>
                <h3 style='margin: 0.5rem 0;'>T.I.P.</h3>
                <p style='margin: 0; font-weight: bold;'>Smart Assistant</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Quick Guide")
        st.markdown("""
        **How to use:**
        1. Type your question
        2. Click **Ask** or use sample questions
        3. Get instant answers from the T.I.P. Manual
        4. Rate the answer quality
        
        **Available Sections:**
        """)
        
        for section in all_sections:
            st.markdown(f"• {section}")
        
        st.divider()
        
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 Total Chunks", len(chunks))
        with col2:
            st.metric("📑 Sections", len(all_sections))
    
    # ========================================================================
    # MAIN CONTENT - REACT-STYLE COMPONENTS
    # ========================================================================
    
    # Initialize session state for React-like navigation
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # HEADER SECTION
    st.markdown(f"""
    <div class="header-section">
        <div class="main-title">Smartual Student Assistant</div>
        <div class="subtitle">🏫 Technological Institute of the Philippines</div>
    </div>
    """, unsafe_allow_html=True)
    
    # HOME PAGE (React-style conditional rendering)
    if st.session_state.current_answer is None:
        render_home_page(model, chunks, index, chunk_embeds, section_examples, all_sections)
    else:
        render_results_page()
    
    # FOOTER
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {SECONDARY}; padding: 2rem 0 1rem 0;">
        <p style="margin: 0.2rem; font-size: 0.9rem;">© 2025 Smart Student Assistant | T.I.P. Q&A System</p>
        <p style="margin: 0.2rem; font-size: 0.8rem;">Powered by AI • Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

def render_home_page(model, chunks, index, chunk_embeds, section_examples, all_sections):
    """Render the home page component (React-style)"""
    
    # Welcome Section with School Logo
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            st.image("TIP LOGO.jpg", width=150)
        except:
            st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h1 style='font-size: 4rem; margin: 0;'>🎓</h1>
                <p style='font-weight: bold; margin: 0;'>T.I.P.</p>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='padding: 1rem;'>
            <h2 style='margin-bottom: 1rem; color: {TEXT};'>👋 Welcome to Your Smart Assistant!</h2>
            <p style='font-size: 1.2rem; margin-bottom: 0; color: {SECONDARY};'>
            Get instant answers to your questions about the T.I.P. Student Manual
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Search Section
    st.markdown(f"""
    <div class="search-section">
        <h3 style='color: {TEXT}; margin-bottom: 1.5rem;'>💬 Ask Your Question</h3>
    </div>
    """, unsafe_allow_html=True)
        
    # Input Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., How can I apply for a scholarship? What are the grading policies?",
            label_visibility="collapsed",
            key="main_question_input"
        )
    
    with col2:
        ask_pressed = st.button("🚀 **ASK**", type="primary", use_container_width=True)
    
    # Sample Questions
    st.markdown("### 💡 Sample Questions")
    sample_cols = st.columns(2)
    
    samples = [
        "What are the admission requirements for T.I.P.?",
        "How is the final grade computed in courses?",
        "What scholarships are available for students?",
        "What is the policy on academic probation?",
        "How many absences are allowed per semester?",
        "What services does the T.I.P. library offer?",
        "How can I request for official documents?",
        "What are the guidelines for thesis writing?"
    ]
    
    for i, sample in enumerate(samples):
        with sample_cols[i % 2]:
            if st.button(f"📌 {sample}", key=f"sample_{i}", use_container_width=True):
                st.session_state.current_question = sample
                process_question(sample, model, chunks, index, chunk_embeds, section_examples)
                st.rerun()
    
    # Manual ask processing
    if ask_pressed and question.strip():
        st.session_state.current_question = question
        process_question(question, model, chunks, index, chunk_embeds, section_examples)
        st.rerun()
    elif ask_pressed:
        st.warning("⚠️ Please enter a question first!")

def render_results_page():
    """Render the results page component (React-style)"""
    
    # Back button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.current_answer = None
            st.rerun()
    
    # Display Results
    answer_data = st.session_state.current_answer
    
    st.markdown(f"""
    <div class="answer-card">
        <h3 style='color: {TEXT}; margin-top: 0; border-bottom: 2px solid {PRIMARY}; padding-bottom: 1rem;'>
            💡 Answer
        </h3>
        <div class="answer-highlight">
            {answer_data['answer']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    st.markdown("### 📊 Answer Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style='font-size: 2rem;'>📂</div>
            <h4 style='margin: 0.5rem 0;'>Section</h4>
            <p style='margin: 0; font-size: 1.1rem;'><strong>{answer_data['section']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence_color = SUCCESS if answer_data['confidence'] > 0.6 else WARNING
        st.markdown(f"""
        <div class="metric-card">
            <div style='font-size: 2rem; color: {confidence_color};'>💯</div>
            <h4 style='margin: 0.5rem 0;'>Confidence</h4>
            <p style='margin: 0; font-size: 1.1rem;'><strong>{answer_data['confidence']:.2%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style='font-size: 2rem;'>📄</div>
            <h4 style='margin: 0.5rem 0;'>Source</h4>
            <p style='margin: 0; font-size: 1.1rem;'><strong>T.I.P. Manual</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Source Details
    with st.expander("🔍 View Source Information", expanded=False):
        for i, (chunk, score) in enumerate(zip(answer_data['top_chunks'], answer_data['similarities']), 1):
            st.markdown(f"""
            **Source {i}** (Relevance: `{score:.2%}`) - **Section:** *{chunk['section']}*
            
            {chunk['chunk_text']}
            """)
            if i < len(answer_data['top_chunks']):
                st.divider()
    
    # Feedback Section
    st.markdown("---")
    st.markdown("### 📣 Rate This Answer")
    
    feedback_col1, feedback_col2 = st.columns(2)
    
    with feedback_col1:
        if st.button("👍 Helpful Answer", use_container_width=True, type="primary"):
            save_feedback(
                st.session_state.current_question,
                answer_data['answer'],
                answer_data['section'],
                answer_data['confidence'],
                True
            )
            st.success("🎉 Thank you for your feedback!")
            st.balloons()
    
    with feedback_col2:
        if st.button("👎 Needs Improvement", use_container_width=True):
            save_feedback(
                st.session_state.current_question,
                answer_data['answer'],
                answer_data['section'],
                answer_data['confidence'],
                False
            )
            st.info("📝 Thanks for helping us improve!")

def process_question(question, model, chunks, index, chunk_embeds, section_examples):
    """Process question and store results in session state"""
    with st.spinner("🔍 Searching through the Student Manual..."):
        # Classify question to section
        pred_section, section_conf = classify_question(question, model, section_examples)
        
        # Retrieve top chunks
        top_chunks, similarities = retrieve_chunks(question, model, chunks, index, chunk_embeds, top_k=3)
        
        # Generate answer from best chunk
        best_chunk = top_chunks[0]
        answer, confidence = generate_answer(question, best_chunk, model)
        
        # Store in session state
        st.session_state.current_answer = {
            'answer': answer,
            'section': pred_section,
            'confidence': confidence,
            'top_chunks': top_chunks,
            'similarities': similarities
        }

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
